"""Experimental: grow the calibration automatically once it is roughly right.

For every aircraft predicted inside the frame, look in a small window around
the prediction for a compact dark-on-sky (day) or bright (night, nav lights)
blob. A detection only becomes a calibration point when the offset between
detection and prediction stays stable over several looks *while the aircraft
moves across the image* - a static feature (tree, pylon, cloud edge) gives an
offset that drifts with the aircraft, so it is rejected.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque

import cv2
import numpy as np

log = logging.getLogger("adsb_camera.autodetect")


class AutoRefiner:
    LOOK_EVERY_S = 0.5
    NEED_LOOKS = 4
    STABLE_PX = 3.0
    MIN_TRAVEL_PX = 6.0

    def __init__(self, interval_s=10.0, search_px=45, min_score=7.0):
        self.interval_s = interval_s
        self.search_px = search_px
        self.min_score = min_score
        self._last_look = 0.0
        self._hist = defaultdict(lambda: deque(maxlen=8))  # icao -> (t, pred, det)
        self._last_added: dict = {}
        self.stats = {"looks": 0, "detections": 0, "accepted": 0}

    def process(self, frame: np.ndarray, t: float, scene: list, max_offset_px: float = 1e9):
        """Returns [(icao, (u, v) native px), ...] accepted this call.
        Detections further than max_offset_px from the prediction are ignored:
        with a good calibration a big offset means a wrong blob or a frame
        whose timestamp doesn't match its content (stream backlog)."""
        now = time.time()
        if now - self._last_look < self.LOOK_EVERY_S:
            return []
        self._last_look = now
        h, w = frame.shape[:2]
        R = int(self.search_px * max(1.0, w / 1920))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        views = [a for a in scene if a["in_view"] and a["uv"] is not None and not a["on_ground"]]
        out = []
        for a in views:
            u, v = a["uv"]
            if not (R <= u < w - R and R <= v < h - R):
                continue
            # Ambiguous if another aircraft is predicted nearby.
            if any(b is not a and np.hypot(b["uv"][0] - u, b["uv"][1] - v) < 2.5 * R for b in views):
                continue
            det = self._detect(gray, int(u), int(v), R)
            self.stats["looks"] += 1
            if det is None:
                continue
            if np.hypot(det[0] - u, det[1] - v) > max_offset_px:
                self.stats.setdefault("rejected_far", 0)
                self.stats["rejected_far"] += 1
                continue
            self.stats["detections"] += 1
            hist = self._hist[a["icao"]]
            hist.append((t, (u, v), det))
            ok = self._stable(hist)
            if ok is None:
                continue
            if now - self._last_added.get(a["icao"], 0.0) < self.interval_s:
                continue
            self._last_added[a["icao"]] = now
            self.stats["accepted"] += 1
            out.append((a["icao"], det))
        return out

    def _detect(self, gray, u, v, R):
        roi = gray[v - R:v + R, u - R:u + R]
        if roi.size == 0:
            return None
        k = max(5, (R // 4) | 1)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        resp = cv2.max(cv2.morphologyEx(roi, cv2.MORPH_BLACKHAT, ker),
                       cv2.morphologyEx(roi, cv2.MORPH_TOPHAT, ker))
        resp = cv2.GaussianBlur(resp, (3, 3), 0).astype(np.float32)
        _, peak, _, loc = cv2.minMaxLoc(resp)
        med = float(np.median(resp))
        mad = float(np.median(np.abs(resp - med))) * 1.4826 + 1.0
        if (peak - med) / mad < self.min_score or peak < 10:
            return None
        # Reject if there's a second, similar peak (texture, not a lone aircraft).
        mask = resp.copy()
        cv2.circle(mask, loc, k, 0, -1)
        if cv2.minMaxLoc(mask)[1] > 0.7 * peak:
            return None
        # Sub-pixel centroid around the peak.
        x0, y0 = loc
        win = resp[max(0, y0 - 2):y0 + 3, max(0, x0 - 2):x0 + 3]
        ys, xs = np.mgrid[max(0, y0 - 2):max(0, y0 - 2) + win.shape[0],
                          max(0, x0 - 2):max(0, x0 - 2) + win.shape[1]]
        wsum = float(win.sum()) or 1.0
        cx, cy = float((xs * win).sum() / wsum), float((ys * win).sum() / wsum)
        return (u - R + cx, v - R + cy)

    def _stable(self, hist):
        if len(hist) < self.NEED_LOOKS:
            return None
        recent = list(hist)[-self.NEED_LOOKS:]
        if recent[-1][0] - recent[0][0] > self.NEED_LOOKS * self.LOOK_EVERY_S * 3:
            return None  # looks too spread out
        offs = np.array([[d[0] - p[0], d[1] - p[1]] for _, p, d in recent])
        if np.max(np.linalg.norm(offs - np.median(offs, axis=0), axis=1)) > self.STABLE_PX:
            return None
        travel = np.hypot(recent[-1][1][0] - recent[0][1][0], recent[-1][1][1] - recent[0][1][1])
        if travel < self.MIN_TRAVEL_PX:
            return None
        return recent[-1][2]
