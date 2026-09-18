"""Threaded video capture that always holds the newest frame.

Accepts anything OpenCV/FFmpeg can open: an RTSP URL (IPyCam main stream),
an MJPEG URL (IPyCam http://host:8080/stream.mjpeg), a video file, or a local
camera index (0, 1, ...). Each frame is stamped with time.time() on receipt;
the constant delay between capture and receipt is absorbed by the
calibration's sync offset.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("adsb_camera.video")


def _parse_source(src):
    if isinstance(src, int):
        return src
    s = str(src).strip()
    return int(s) if s.isdigit() else s


class VideoSource:
    def __init__(self, source, rtsp_transport: str = "tcp", reconnect_s: float = 3.0,
                 width: Optional[int] = None, height: Optional[int] = None):
        self.source = _parse_source(source)
        self.rtsp_transport = rtsp_transport
        self.reconnect_s = reconnect_s
        self.req_size = (width, height)
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._t = 0.0
        self._seq = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._times: deque = deque(maxlen=60)
        self.connected = False
        self.error = ""
        self.is_file = isinstance(self.source, str) and os.path.isfile(self.source)

    # ── public ────────────────────────────────────────────────────────────
    def start(self):
        self._thread = threading.Thread(target=self._run, name="video", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def latest(self):
        """(frame, receive_time, seq) - frame is shared, don't modify it."""
        with self._lock:
            return self._frame, self._t, self._seq

    @property
    def fps(self) -> float:
        with self._lock:
            ts = list(self._times)
        if len(ts) < 2 or ts[-1] - ts[0] <= 0:
            return 0.0
        return (len(ts) - 1) / (ts[-1] - ts[0])

    def describe(self) -> str:
        s = str(self.source)
        if "@" in s:  # hide credentials in rtsp://user:pass@host
            scheme, rest = s.split("://", 1) if "://" in s else ("", s)
            s = f"{scheme}://***@{rest.split('@', 1)[1]}"
        return s

    # ── worker ────────────────────────────────────────────────────────────
    def _open(self):
        src = self.source
        if isinstance(src, str) and src.lower().startswith("rtsp"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{self.rtsp_transport}"
        if isinstance(src, int):
            cap = cv2.VideoCapture(src)
            w, h = self.req_size
            if w and h:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(w))
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h))
        else:
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                cap = cv2.VideoCapture(src)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def _run(self):
        while not self._stop.is_set():
            log.info("Video: opening %s", self.describe())
            cap = self._open()
            if not cap.isOpened():
                self.connected = False
                self.error = f"cannot open {self.describe()}"
                log.warning("Video: %s; retrying in %.0fs", self.error, self.reconnect_s)
                self._stop.wait(self.reconnect_s)
                continue
            file_fps = cap.get(cv2.CAP_PROP_FPS) if self.is_file else 0
            self.connected, self.error = True, ""
            failures = 0
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    if self.is_file:  # loop files (handy for testing)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    failures += 1
                    if failures > 25:
                        break
                    time.sleep(0.02)
                    continue
                failures = 0
                now = time.time()
                with self._lock:
                    self._frame, self._t = frame, now
                    self._seq += 1
                    self._times.append(now)
                if file_fps:
                    time.sleep(1.0 / max(1.0, file_fps))
            cap.release()
            self.connected = False
            if not self._stop.is_set():
                self.error = "stream lost"
                log.warning("Video: stream lost; reconnecting in %.0fs", self.reconnect_s)
                self._stop.wait(self.reconnect_s)
