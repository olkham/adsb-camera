"""A camera whose auto exposure gets stuck should be nudged, then reopened.

A webcam left running for hours can settle on an exposure and stop adapting, so the
picture goes dark in broad daylight. The watchdog samples the brightness, and when it
drops it re-runs the camera's auto exposure; if that does not help it asks for the
camera to be reopened.

Nothing can tell a stuck exposure from a scene that is honestly dark, so the escalation
has to stop by itself - otherwise every night would be spent reconnecting.

Run:  python -m pytest -q      (or: python tests/test_exposure_watchdog.py)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from adsb_camera.video import SETTLE_S, VideoSource


class FakeCap:
    """A camera whose exposure can be stuck, and freed by re-running auto exposure."""

    def __init__(self, stuck=False, freeable=True):
        self.stuck, self.freeable = stuck, freeable
        self.ae_calls = []

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_AUTO_EXPOSURE:
            self.ae_calls.append(value)
            if len(self.ae_calls) >= 2 and self.freeable:
                self.stuck = False          # off-then-on freed it
        return True

    def level(self):
        return 8 if self.stuck else 120


def source(**kw):
    kw.setdefault("exposure_check_s", 3600)
    v = VideoSource(1, **kw)
    v.backend = "dshow"
    return v


def feed(v, cap, seconds, start=1000.0, step=1.0):
    """Run the sampling and the watchdog as the capture loop would. Returns reconnects."""
    reconnects = []
    for i in range(int(seconds / step)):
        now = start + i * step
        v._sample_brightness(np.full((90, 160, 3), cap.level(), np.uint8))
        if v._watchdog(cap, now):
            reconnects.append(now)
    return reconnects


def test_a_bright_picture_is_left_alone():
    v = source()
    cap = FakeCap(stuck=False)
    v._next_check = 1000.0
    assert feed(v, cap, 7200) == []
    assert cap.ae_calls == [], "the camera was touched although the picture was fine"


def test_a_dark_picture_gets_auto_exposure_re_run():
    v = source()
    cap = FakeCap(stuck=True)
    v._next_check = 1010.0                       # the hourly check comes round
    reconnects = feed(v, cap, 120)
    assert len(cap.ae_calls) >= 2, "auto exposure was not switched off and back on"
    assert reconnects == [], "a nudge that worked should not also reconnect"
    assert v._dark_stage == 0, "should be back to normal once the picture recovered"
    assert v.brightness > 100


def test_a_camera_that_stays_dark_is_reconnected():
    v = source()
    cap = FakeCap(stuck=True, freeable=False)    # nudging will not help this one
    v._next_check = 1010.0
    reconnects = feed(v, cap, 120)
    assert len(cap.ae_calls) >= 2, "it should try the nudge first"
    assert len(reconnects) == 1, f"expected exactly one reconnect, got {len(reconnects)}"
    assert reconnects[0] >= 1010.0 + SETTLE_S, "it judged the nudge before it could take effect"


def test_a_dark_scene_is_not_reconnected_over_and_over():
    """Night looks exactly like a stuck exposure. Try once, then leave the camera be."""
    v = source(exposure_check_s=60)
    cap = FakeCap(stuck=True, freeable=False)
    v._next_check = 1010.0
    reconnects = feed(v, cap, 8 * 3600, step=5.0)     # eight hours of darkness
    assert len(reconnects) == 1, f"reconnected {len(reconnects)} times during one dark spell"
    assert len(cap.ae_calls) == 2, f"nudged auto exposure {len(cap.ae_calls)} times"


def test_it_arms_again_once_the_picture_comes_back():
    v = source(exposure_check_s=60)
    cap = FakeCap(stuck=True, freeable=False)
    v._next_check = 1010.0
    first = feed(v, cap, 3600, start=1000.0, step=5.0)
    assert len(first) == 1

    cap.stuck = False                              # daylight
    feed(v, cap, 600, start=5000.0, step=5.0)
    assert v._dark_stage == 0, "should have re-armed once the picture recovered"

    cap.stuck, cap.freeable = True, False          # and it goes wrong again
    again = feed(v, cap, 3600, start=6000.0, step=5.0)
    assert len(again) == 1, "a new dark spell should be acted on afresh"


def test_the_check_can_be_turned_off():
    v = source(exposure_check_s=0)
    assert v.check_s == 0
    # The capture loop only arms the watchdog when check_s > 0 and the source is a camera.
    assert not (v.check_s > 0 and isinstance(v.source, int))


def test_brightness_is_averaged_not_taken_from_one_frame():
    v = source()
    for level in (120, 120, 120, 0):               # one black frame, e.g. a shutter blink
        v._sample_brightness(np.full((90, 160, 3), level, np.uint8))
    assert v.brightness > 80, f"one odd frame moved the average to {v.brightness}"


def test_streams_and_files_are_not_watched():
    """Only a local camera has exposure controls to nudge."""
    v = VideoSource("rtsp://host/stream", exposure_check_s=3600)
    assert not isinstance(v.source, int)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
