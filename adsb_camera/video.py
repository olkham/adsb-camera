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


BACKENDS = {"auto": None, "any": cv2.CAP_ANY, "dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF,
            "v4l2": cv2.CAP_V4L2, "avfoundation": cv2.CAP_AVFOUNDATION,
            "gstreamer": cv2.CAP_GSTREAMER, "ffmpeg": cv2.CAP_FFMPEG}

# Turning auto exposure off and on again makes the driver run it afresh. The values
# are the backend's own: DirectShow and Media Foundation use these, V4L2 uses 1 and 3.
AE_MANUAL, AE_AUTO = 0.25, 0.75
AE_MANUAL_V4L2, AE_AUTO_V4L2 = 1.0, 3.0
SETTLE_S = 8.0          # give the camera this long to re-expose before judging it
BRIGHT_EVERY_S = 1.0    # how often the picture's brightness is sampled
BRIGHT_SAMPLES = 10     # ... and how many of those are averaged


def _fourcc_name(v) -> str:
    v = int(v or 0)
    return "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4)).strip() if v else ""


class VideoSource:
    def __init__(self, source, rtsp_transport: str = "tcp", reconnect_s: float = 3.0,
                 width: Optional[int] = None, height: Optional[int] = None,
                 fps: Optional[float] = None, fourcc: str = "MJPG", backend: str = "auto",
                 exposure_check_s: float = 3600.0, dark_level: float = 40.0):
        self.source = _parse_source(source)
        self.rtsp_transport = rtsp_transport
        self.reconnect_s = reconnect_s
        self.req_size = (width, height)
        self.req_fps = float(fps or 0)
        self.req_fourcc = (fourcc or "").strip().upper()
        self.backend = (backend or "auto").strip().lower()
        self.format = ""        # what the camera actually gave us, e.g. "MJPG 1920x1080"
        self.check_s = float(exposure_check_s or 0)
        self.dark_level = float(dark_level)
        self.brightness: Optional[float] = None   # average of the picture, 0-255
        self._bright: deque = deque(maxlen=BRIGHT_SAMPLES)
        self._bright_t = 0.0
        self._dark_stage = 0    # 0 all well, 1 auto exposure nudged, 2 tried everything
        self._dark_deadline = 0.0
        self._next_check = 0.0
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
    def _open_camera(self, src: int):
        """A local webcam, asking for a compressed format so it runs at full speed.

        USB webcams default to uncompressed frames (YUY2), and the bus cannot carry
        many of those: a 1080p webcam drops to about 5 fps. Asking for MJPG fixes it,
        but only if the request reaches the driver while the format is being
        negotiated - setting it after opening is ignored unless the frame size was
        set first. Opening with the properties does it in one go; the rest is
        fallbacks for cameras or backends that refuse them.
        """
        api = BACKENDS.get(self.backend) or cv2.CAP_ANY
        w, h = self.req_size
        cc = cv2.VideoWriter_fourcc(*self.req_fourcc[:4]) if len(self.req_fourcc) == 4 else 0
        params = self._capture_params(w, h, cc)

        if params:
            cap = cv2.VideoCapture(src, api, params)
            if cap.isOpened():
                return cap
            log.info("Video: camera %s would not open with the requested format; "
                     "setting it afterwards instead", src)
            cap.release()

        cap = cv2.VideoCapture(src, api)
        if cap.isOpened():
            if w and h:                # size first, then format: the order matters here
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(w))
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h))
            if cc:
                cap.set(cv2.CAP_PROP_FOURCC, cc)
            if self.req_fps:
                cap.set(cv2.CAP_PROP_FPS, self.req_fps)
        return cap

    def _capture_params(self, w, h, cc) -> list:
        params = []
        if cc:
            params += [cv2.CAP_PROP_FOURCC, cc]
        if w and h:
            params += [cv2.CAP_PROP_FRAME_WIDTH, int(w), cv2.CAP_PROP_FRAME_HEIGHT, int(h)]
        if self.req_fps:
            params += [cv2.CAP_PROP_FPS, int(self.req_fps)]
        return params

    def _open(self):
        src = self.source
        if isinstance(src, str) and src.lower().startswith("rtsp"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{self.rtsp_transport}"
        if isinstance(src, int):
            cap = self._open_camera(src)
        else:
            api = BACKENDS.get(self.backend) or cv2.CAP_FFMPEG
            cap = cv2.VideoCapture(src, api)
            if not cap.isOpened():
                cap = cv2.VideoCapture(src)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def _describe_format(self, cap) -> str:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cc = _fourcc_name(cap.get(cv2.CAP_PROP_FOURCC))
        return " ".join(x for x in (cc, f"{w}x{h}" if w and h else "") if x)

    # ── exposure watchdog ─────────────────────────────────────────────────
    def _sample_brightness(self, frame):
        """Average brightness of the picture, smoothed over the last few seconds."""
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        self._bright.append(float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean()))
        self.brightness = sum(self._bright) / len(self._bright)

    def _kick_auto_exposure(self, cap) -> bool:
        """Switch auto exposure off and straight back on, so the driver runs it afresh."""
        manual, auto = (AE_MANUAL_V4L2, AE_AUTO_V4L2) if self.backend == "v4l2"             else (AE_MANUAL, AE_AUTO)
        try:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, manual)
            time.sleep(0.2)
            return bool(cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto))
        except cv2.error as e:
            log.warning("Video: could not reach the camera's auto exposure: %s", e)
            return False

    def _watchdog(self, cap, now) -> bool:
        """Catch a camera whose auto exposure has got stuck. True asks for a reconnect.

        A webcam left running for hours can settle on an exposure and stop adapting,
        leaving the picture dark in broad daylight. Nudging auto exposure usually
        frees it; reopening the camera is the bigger hammer. Neither can tell a stuck
        exposure from a scene that is honestly dark, so once both have been tried we
        leave it alone until the picture brightens of its own accord - otherwise every
        night would be spent reconnecting.
        """
        if self._dark_stage == 1:                  # nudged; has it taken effect?
            if now < self._dark_deadline:
                return False
            self._next_check = now + self.check_s
            b = self.brightness or 0.0
            if b >= self.dark_level:
                log.info("Video: auto exposure recovered the picture (brightness %.0f)", b)
                self._dark_stage = 0
                return False
            log.warning("Video: still dark (brightness %.0f) after re-running auto "
                        "exposure; reconnecting the camera", b)
            self._dark_stage = 2
            return True

        if now < self._next_check:
            return False
        self._next_check = now + self.check_s
        b = self.brightness
        if b is None or len(self._bright) < 3:
            return False
        if b >= self.dark_level:
            if self._dark_stage:
                log.info("Video: picture is back to normal (brightness %.0f)", b)
            self._dark_stage = 0
            return False
        if self._dark_stage == 2:                  # already tried both; it may just be night
            log.info("Video: picture still dark (brightness %.0f); leaving the camera "
                     "alone until it brightens", b)
            return False
        log.warning("Video: picture is dark (brightness %.0f, below %.0f) - re-running "
                    "the camera's auto exposure", b, self.dark_level)
        self._kick_auto_exposure(cap)
        self._bright.clear()                       # judge it on what it sends from now on
        self._dark_stage = 1
        self._dark_deadline = now + SETTLE_S
        return False

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
            self.format = self._describe_format(cap)
            if isinstance(self.source, int):
                log.info("Video: camera %s giving %s", self.source, self.format or "?")
                if self.req_fourcc and _fourcc_name(cap.get(cv2.CAP_PROP_FOURCC)) != self.req_fourcc:
                    log.warning("Video: asked for %s but got %s - an uncompressed format will "
                                "limit the frame rate", self.req_fourcc, self.format or "?")
            self.connected, self.error = True, ""
            failures = 0
            # Only a live camera can have its exposure nudged; files and streams cannot.
            watch = bool(self.check_s > 0 and isinstance(self.source, int))
            reopen = False
            self._bright.clear()
            self.brightness = None
            self._next_check = time.time() + self.check_s
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
                if watch and now - self._bright_t >= BRIGHT_EVERY_S:
                    self._bright_t = now
                    self._sample_brightness(frame)
                    if self._watchdog(cap, now):
                        reopen = True
                        break
                if file_fps:
                    time.sleep(1.0 / max(1.0, file_fps))
            cap.release()
            self.connected = False
            if self._stop.is_set():
                continue
            if reopen:
                log.info("Video: reopening the camera")
                self._stop.wait(1.0)
            else:
                self.error = "stream lost"
                log.warning("Video: stream lost; reconnecting in %.0fs", self.reconnect_s)
                self._stop.wait(self.reconnect_s)
