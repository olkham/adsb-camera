"""Saved frames for later calibration.

A "saved frame" is a snapshot of the video plus the aircraft data needed to
identify what's in it, persisted to disk so points can be added long after
the aircraft has left the picture (unlike freeze/unfreeze, which only holds
one frame in memory). Frames can be saved manually or by the auto-log
feature, which watches for tracks entering the field of view.

Each frame is a pair of files: ``<id>.jpg`` (the raw frame, no overlay) and
``<id>.json`` (frame time + aircraft snapshot), alongside ``<id>_thumb.jpg``,
a small copy used by the timeline strip in the UI. A lightweight
``index.json`` lists every frame without its snapshot, for fast browsing.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
import uuid
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("adsb_camera.logbook")

THUMB_W = 240          # px wide; enough for the timeline strip at 2x


class LogStore:
    def __init__(self, dir_path: str, max_auto: int = 300):
        self.dir = dir_path
        self.max_auto = max_auto
        os.makedirs(dir_path, exist_ok=True)
        self._index_path = os.path.join(dir_path, "index.json")
        self._lock = threading.RLock()
        self._entries: list = self._load_index()

    def _load_index(self) -> list:
        if not os.path.exists(self._index_path):
            return []
        try:
            with open(self._index_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError) as e:
            log.error("Could not read %s: %s", self._index_path, e)
            return []

    def _save_index(self):
        fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(self._entries, fh, indent=1)
        os.replace(tmp, self._index_path)

    # ── write ─────────────────────────────────────────────────────────────
    def add(self, frame: np.ndarray, t: float, snapshot: dict, source: str = "manual",
            reason: str = "") -> dict:
        fid = f"{int(t * 1000)}_{uuid.uuid4().hex[:6]}"
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise RuntimeError("could not encode frame")
        h, w = frame.shape[:2]
        with open(os.path.join(self.dir, f"{fid}.jpg"), "wb") as fh:
            fh.write(buf.tobytes())
        with open(os.path.join(self.dir, f"{fid}.json"), "w", encoding="utf-8") as fh:
            json.dump({"t": t, "snapshot": snapshot}, fh)
        self._write_thumb(fid, frame)
        entry = {"id": fid, "t": t, "w": w, "h": h, "source": source, "reason": reason,
                 "created": time.time(), "n_aircraft": len(snapshot)}
        with self._lock:
            self._entries.append(entry)
            self._save_index()
            if source == "auto":
                self._prune_auto()
        return entry

    def _write_thumb(self, fid: str, frame: np.ndarray) -> Optional[str]:
        h, w = frame.shape[:2]
        s = THUMB_W / float(w)
        small = cv2.resize(frame, (THUMB_W, max(1, int(round(h * s)))),
                           interpolation=cv2.INTER_AREA) if s < 1 else frame
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            log.warning("Could not encode thumbnail for %s", fid)
            return None
        path = self.thumb_path(fid)
        with open(path, "wb") as fh:
            fh.write(buf.tobytes())
        return path

    def _prune_auto(self):
        """Drop the oldest auto-logged frames beyond the configured cap (manual saves are never pruned)."""
        auto = sorted((e for e in self._entries if e["source"] == "auto"), key=lambda e: e["t"])
        excess = len(auto) - self.max_auto
        if excess <= 0:
            return
        drop = {e["id"] for e in auto[:excess]}
        for fid in drop:
            self._remove_files(fid)
        self._entries = [e for e in self._entries if e["id"] not in drop]
        self._save_index()

    # ── read ──────────────────────────────────────────────────────────────
    def list(self) -> list:
        with self._lock:
            return sorted(self._entries, key=lambda e: e["t"], reverse=True)

    def get(self, fid: str) -> Optional[dict]:
        with self._lock:
            return next((e for e in self._entries if e["id"] == fid), None)

    def get_full(self, fid: str) -> Optional[dict]:
        """Frame time + aircraft snapshot (not cached; read from disk each time)."""
        if not self.get(fid):
            return None
        path = os.path.join(self.dir, f"{fid}.json")
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError) as e:
            log.error("Could not read %s: %s", path, e)
            return None

    def load_image(self, fid: str) -> Optional[np.ndarray]:
        path = self.frame_path(fid)
        if not os.path.exists(path):
            return None
        return cv2.imread(path)

    def frame_path(self, fid: str) -> str:
        return os.path.join(self.dir, f"{fid}.jpg")

    def thumb_path(self, fid: str) -> str:
        return os.path.join(self.dir, f"{fid}_thumb.jpg")

    def ensure_thumb(self, fid: str) -> Optional[str]:
        """Path to the thumbnail, making it from the full frame if it isn't there yet."""
        path = self.thumb_path(fid)
        if os.path.exists(path):
            return path
        img = self.load_image(fid)
        if img is None:
            return None
        return self._write_thumb(fid, img)

    # ── delete ────────────────────────────────────────────────────────────
    def _remove_files(self, fid: str):
        for suffix in (".jpg", ".json", "_thumb.jpg"):
            try:
                os.remove(os.path.join(self.dir, f"{fid}{suffix}"))
            except OSError:
                pass

    def delete(self, fid: str) -> bool:
        with self._lock:
            if not self.get(fid):
                return False
            self._entries = [x for x in self._entries if x["id"] != fid]
            self._save_index()
        self._remove_files(fid)
        return True

    def clear(self) -> int:
        """Delete every saved frame. Returns how many were removed."""
        with self._lock:
            ids = [e["id"] for e in self._entries]
            self._entries = []
            self._save_index()
        for fid in ids:
            self._remove_files(fid)
        return len(ids)
