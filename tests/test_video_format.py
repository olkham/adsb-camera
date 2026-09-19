"""A webcam must be asked for a compressed format, in the way the driver listens to.

USB webcams hand over uncompressed frames (YUY2) unless told otherwise, and the bus
cannot carry many of those - a Logitech C920 measured 4.9 fps at 1080p that way, and
29.5 fps once it was asked for MJPG. The catch is that the request only lands if it
reaches the driver while the format is being negotiated: opening with the properties
works, and so does setting the size before the format, but setting the format first
and the size afterwards silently leaves the camera uncompressed.

No camera needed: this drives VideoSource against a stand-in for cv2.VideoCapture and
checks what it was asked for.

Run:  python -m pytest -q      (or: python tests/test_video_format.py)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from adsb_camera import video as vid

MJPG = cv2.VideoWriter_fourcc(*"MJPG")


class FakeCap:
    """Records how it was opened and what was set on it."""

    opened_with = []          # every construction, in order
    instances = []

    def __init__(self, src, api=None, params=None, opens=True):
        self.src, self.api, self.params, self._open = src, api, list(params or []), opens
        self.sets = []
        FakeCap.opened_with.append(self)
        FakeCap.instances.append(self)

    def isOpened(self):
        return self._open

    def set(self, prop, value):
        self.sets.append((prop, value))
        return True

    def get(self, prop):
        return 0

    def release(self):
        pass

    @classmethod
    def reset(cls):
        cls.opened_with, cls.instances = [], []


def patched(monkey_opens):
    """Install a fake cv2.VideoCapture; monkey_opens(params) decides if it opens."""
    def factory(src, api=None, params=None):
        return FakeCap(src, api, params, opens=monkey_opens(params))
    FakeCap.reset()
    vid.cv2.VideoCapture = factory


def restore():
    vid.cv2.VideoCapture = _real


_real = cv2.VideoCapture


def make(**kw):
    kw.setdefault("width", 1920)
    kw.setdefault("height", 1080)
    kw.setdefault("fps", 30)
    return vid.VideoSource(1, **kw)


def test_format_is_requested_while_the_camera_opens():
    patched(lambda params: True)
    try:
        cap = make()._open_camera(1)
        assert cap.params, "the camera was opened without asking for a format"
        assert cap.params[0] == cv2.CAP_PROP_FOURCC and cap.params[1] == MJPG, cap.params
        assert cv2.CAP_PROP_FRAME_WIDTH in cap.params and cv2.CAP_PROP_FPS in cap.params
        assert len(FakeCap.instances) == 1, "should not have needed a second attempt"
    finally:
        restore()


def test_falls_back_to_setting_the_size_before_the_format():
    """Cameras that refuse the properties at open still have to end up compressed."""
    patched(lambda params: not params)      # only the plain open succeeds
    try:
        cap = make()._open_camera(1)
        props = [p for p, _ in cap.sets]
        assert cv2.CAP_PROP_FOURCC in props, "fell back without asking for a format at all"
        assert props.index(cv2.CAP_PROP_FRAME_WIDTH) < props.index(cv2.CAP_PROP_FOURCC), \
            "format was set before the size, which the driver ignores"
    finally:
        restore()


def test_fourcc_can_be_turned_off():
    patched(lambda params: True)
    try:
        cap = make(fourcc="")._open_camera(1)
        assert cv2.CAP_PROP_FOURCC not in cap.params, "asked for a format when told not to"
        assert cv2.CAP_PROP_FRAME_WIDTH in cap.params, "should still ask for the size"
    finally:
        restore()


def test_backend_choice_is_passed_through():
    patched(lambda params: True)
    try:
        assert make(backend="dshow")._open_camera(1).api == cv2.CAP_DSHOW
        assert make(backend="msmf")._open_camera(1).api == cv2.CAP_MSMF
        assert make(backend="auto")._open_camera(1).api == cv2.CAP_ANY
    finally:
        restore()


def test_fourcc_name_round_trip():
    assert vid._fourcc_name(MJPG) == "MJPG"
    assert vid._fourcc_name(0) == ""


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
