"""The rendered image must always match the frame that is loaded.

Loading a saved frame sets ``core.frozen`` and bumps ``core.version``; the render
loop skips redrawing while the version it last drew is still current. If those two
are read at different moments, a load that lands in between makes the loop stamp
the new version onto the old picture - and then never redraw it, so the viewer sits
on the previous frame (with fresh aircraft data beside it) until something else
bumps the version. That is a calibration hazard: points get clicked on the wrong
picture.

Run:  python -m pytest -q      (or: python tests/test_render_freshness.py)
"""
import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adsb_camera.app import AdsbCamera
from adsb_camera.config import load_config

DARK, BRIGHT = 20, 230          # "night" and "daylight" frames


class LockProxy:
    """Stands in for core.lock and runs a hook on the n-th acquisition."""

    def __init__(self, lock):
        self.lock = lock
        self.hook = None
        self.at = 0
        self.n = 0

    def arm(self, at, hook):
        self.n, self.at, self.hook = 0, at, hook

    def __enter__(self):
        self.n += 1
        if self.hook and self.n == self.at:
            hook, self.hook = self.hook, None   # once only; the hook re-enters this lock
            hook()
        return self.lock.__enter__()

    def __exit__(self, *exc):
        return self.lock.__exit__(*exc)


def make_core(tmp):
    cfg = load_config(None)
    cfg["data_dir"] = tmp
    core = AdsbCamera(cfg)
    core.clients = 1                 # otherwise rendering is skipped as nobody is watching
    return core


def add_frame(core, level, t):
    img = np.full((360, 640, 3), level, np.uint8)
    return core.logstore.add(img, t, {}, source="manual")["id"]


def published_level(core):
    jpg, _ = core.latest_jpeg()
    assert jpg is not None, "nothing has been published"
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    return float(img.mean())


def test_view_then_render_shows_that_frame():
    with tempfile.TemporaryDirectory() as tmp:
        core = make_core(tmp)
        night = add_frame(core, DARK, 1000.0)
        day = add_frame(core, BRIGHT, 2000.0)

        for fid, level in ((night, DARK), (day, BRIGHT), (night, DARK)):
            assert core.view_frame(fid)["ok"]
            core._render_once()
            assert abs(published_level(core) - level) < 25, f"wrong frame published for {fid}"


def test_frame_loaded_mid_render_is_not_skipped():
    """A load landing between the loop's two reads must still reach the screen."""
    with tempfile.TemporaryDirectory() as tmp:
        core = make_core(tmp)
        night = add_frame(core, DARK, 1000.0)
        day = add_frame(core, BRIGHT, 2000.0)

        core.view_frame(night)
        core._render_once()
        assert abs(published_level(core) - DARK) < 25

        # Switch to the daylight frame just after the loop has picked up the night one.
        proxy = LockProxy(core.lock)
        core.lock = proxy
        proxy.arm(2, lambda: core.view_frame(day))
        core._render_once()
        core.lock = proxy.lock

        core._render_once()             # the loop's next pass must catch up
        assert core.frozen["t"] == 2000.0
        assert abs(published_level(core) - BRIGHT) < 25, \
            "viewer is stuck on the previous frame after loading a new one"


def test_repeated_renders_do_not_redraw_an_unchanged_frame():
    """The version stamp must still suppress pointless re-encoding while frozen.

    The first render of a frame whose size differs from the camera model adapts the
    model, which is itself a change and costs one more pass; after that the picture
    must settle.
    """
    with tempfile.TemporaryDirectory() as tmp:
        core = make_core(tmp)
        core.view_frame(add_frame(core, DARK, 1000.0))
        for _ in range(4):
            core._render_once()
        _, seq = core.latest_jpeg()
        for _ in range(5):
            core._render_once()
        assert core.latest_jpeg()[1] == seq, "frozen frame was re-encoded with nothing changed"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
