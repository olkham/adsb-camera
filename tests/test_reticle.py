"""The reticle size setting has to actually change the marker on the frame.

Run:  python -m pytest -q      (or: python tests/test_reticle.py)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adsb_camera.overlay import RETICLE_SHAPES, draw_aircraft

AIRCRAFT = {"uv": (200.0, 200.0), "in_view": True, "rgb": (255, 255, 255), "emergency": False,
            "range_km": 10.0, "trail": [], "callsign": "TEST", "icao": "abc123",
            "alt_ft": 10000, "gs_kt": 300.0, "vrate_fpm": 0.0}


def marker_size(size=None, shape="circle"):
    """Width of the drawn marker, in pixels."""
    img = np.zeros((400, 400, 3), np.uint8)
    opts = {"reticle": True, "reticle_shape": shape, "labels": False,
            "trails": False, "offscreen": False}
    if size is not None:
        opts["reticle_size"] = size
    draw_aircraft(img, [dict(AIRCRAFT)], s=1.0, ui=1.0, opts=opts)
    xs = np.nonzero(img.any(axis=(0, 2)))[0]
    assert len(xs), "nothing was drawn"
    return int(xs[-1] - xs[0])


def test_bigger_setting_draws_a_bigger_reticle():
    small, normal, big = marker_size(0.5), marker_size(1.0), marker_size(2.0)
    assert small < normal < big, (small, normal, big)
    # Twice the setting is about twice across, give or take the outline.
    assert 1.7 < big / normal < 2.3, f"{normal} -> {big} px"


def test_default_is_unchanged_when_the_setting_is_absent():
    assert marker_size(None) == marker_size(1.0)


def test_every_shape_follows_the_setting():
    for shape in RETICLE_SHAPES:
        normal, big = marker_size(1.0, shape), marker_size(2.5, shape)
        assert big > normal * 1.8, f"{shape}: {normal} -> {big} px"


def test_extreme_values_are_clamped_not_crashed():
    assert marker_size(99) == marker_size(4.0), "should clamp to the maximum"
    assert marker_size(0.0001) == marker_size(0.3), "should clamp to the minimum"
    assert marker_size("nonsense") == marker_size(1.0), "junk should fall back to normal"


def test_setting_is_validated_on_the_way_in():
    """The API clamps too, so a bad value never reaches the overlay or gets saved."""
    import tempfile

    from adsb_camera.app import AdsbCamera
    from adsb_camera.config import load_config

    with tempfile.TemporaryDirectory() as tmp:
        cfg = load_config(None)
        cfg["data_dir"] = tmp
        core = AdsbCamera(cfg)
        assert core.overlay["reticle_size"] == 1.0

        core.set_settings({"overlay": {"reticle_size": 2.5}})
        assert core.overlay["reticle_size"] == 2.5
        core.set_settings({"overlay": {"reticle_size": 100}})
        assert core.overlay["reticle_size"] == 4.0
        core.set_settings({"overlay": {"reticle_size": -5}})
        assert core.overlay["reticle_size"] == 0.3
        core.set_settings({"overlay": {"reticle_size": "big"}})
        assert core.overlay["reticle_size"] == 0.3, "junk must not change the setting"
        assert core.state()["settings"]["overlay"]["reticle_size"] == 0.3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
