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
import threading
import time
import uuid
from typing import Optional

import cv2
import numpy as np

from .jsonfile import sweep_temp, write_json

log = logging.getLogger("adsb_camera.logbook")

THUMB_W = 240          # px wide; enough for the timeline strip at 2x


class LogStore:
    def __init__(self, dir_path: str, max_auto: int = 300):
        self.dir = os.path.abspath(dir_path)
        self.max_auto = max_auto
        os.makedirs(self.dir, exist_ok=True)
        self._index_path = os.path.join(self.dir, "index.json")
        self._lock = threading.RLock()
        sweep_temp(self.dir)
        self._entries: list = self._load_index()

    def _load_index(self) -> list:
        if not os.path.exists(self._index_path):
            return []
        try:
            with open(self._index_path, "r", encoding="utf-8") as fh:
                entries = json.load(fh)
        except (OSError, ValueError) as e:
            log.error("Could not read %s: %s", self._index_path, e)
            return []
        # An index write that never landed can leave frames listed whose picture is
        # gone. Ignore those rather than offering frames that cannot be opened.
        keep = [e for e in entries if os.path.exists(self.frame_path(e.get("id", "")))]
        if len(keep) != len(entries):
            log.warning("Ignoring %d saved frame(s) whose image is missing",
                        len(entries) - len(keep))
        return keep

    def _save_index(self):
        write_json(self._index_path, self._entries)

    def _commit(self, entries: list) -> bool:
        """Put ``entries`` on disk, and only then make them the list we are working from.

        Frame files are deleted by the caller afterwards, so a write that fails leaves
        the index and the files still agreeing with each other.
        """
        before = self._entries
        self._entries = entries
        try:
            self._save_index()
            return True
        except OSError as e:
            self._entries = before
            log.error("Could not update %s: %s", self._index_path, e)
            return False

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
        # Adding and pruning in one write: at the cap this runs on every auto-logged
        # frame, and the index is the one file everything else wants to read.
        with self._lock:
            entries = self._entries + [entry]
            drop = self._auto_excess(entries) if source == "auto" else set()
            if self._commit([e for e in entries if e["id"] not in drop] if drop else entries):
                for fid in drop:
                    self._remove_files(fid)
            else:
                # The frame is on disk; keep it listed so the next save covers it, and
                # leave the old ones alone rather than deleting them unrecorded. This
                # must never reach the render loop that called us.
                self._entries = entries
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

    def _auto_excess(self, entries: list) -> set:
        """The oldest auto-logged frames beyond the cap (manual saves are never pruned)."""
        auto = sorted((e for e in entries if e["source"] == "auto"), key=lambda e: e["t"])
        excess = len(auto) - self.max_auto
        return {e["id"] for e in auto[:excess]} if excess > 0 else set()

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
            if not self._commit([x for x in self._entries if x["id"] != fid]):
                return False
        self._remove_files(fid)
        return True

    def clear(self) -> int:
        """Delete every saved frame. Returns how many were removed."""
        with self._lock:
            ids = [e["id"] for e in self._entries]
            if not self._commit([]):
                return 0
        for fid in ids:
            self._remove_files(fid)
        return len(ids)
