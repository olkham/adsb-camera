"""Picture-in-picture: a magnified inset of each nearby aircraft in view, cut
from the full-resolution frame and stacked from the chosen corner.

Run:  python -m pytest -q      (or: python tests/test_pip.py)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adsb_camera.overlay import draw_pip  # noqa: E402


def aircraft(u, v, rng=10.0, in_view=True):
    return {"uv": (u, v), "in_view": in_view, "rgb": (255, 255, 255), "emergency": False,
            "range_km": rng, "callsign": "TEST", "icao": "abc123", "alt_ft": 10000,
            "on_ground": False}


def frame_with_dot(u, v, w=1920, h=1080):
    f = np.zeros((h, w, 3), np.uint8)
    f[v - 2:v + 3, u - 2:u + 3] = (0, 255, 0)        # a 5 px green "aircraft"
    return f


def green(img):
    return (img[:, :, 1] > 150) & (img[:, :, 0] < 100) & (img[:, :, 2] < 100)


def test_inset_shows_the_aircraft_magnified_in_the_corner():
    frame = frame_with_dot(500, 600)
    out = np.zeros((540, 960, 3), np.uint8)             # stream at half size
    draw_pip(out, frame, [aircraft(500, 600)], s=0.5, ui=0.75,
             opts={"pip_count": 1, "pip_zoom": 4, "pip_corner": "top-right"})
    ys, xs = np.nonzero(green(out))
    assert len(xs), "the aircraft is not in the inset"
    assert xs.min() > 960 / 2 and ys.max() < 540 / 2, "inset not in the top-right"
    # 5 px at full resolution, shown 4x the stream's half scale: ~10 px across.
    assert 7 <= xs.max() - xs.min() <= 14, xs.max() - xs.min()


def test_count_corner_and_nearest_first():
    frame = frame_with_dot(300, 300)
    out = np.zeros((1080, 1920, 3), np.uint8)
    scene = [aircraft(300, 300, rng=5), aircraft(900, 500, rng=50), aircraft(1200, 700, rng=20),
             aircraft(100, 100, rng=1, in_view=False)]
    draw_pip(out, frame, scene, s=1.0, ui=1.5,
             opts={"pip_count": 2, "pip_zoom": 3, "pip_corner": "bottom-left"})
    drawn = out.any(axis=2)
    ys, xs = np.nonzero(drawn)
    assert xs.max() < 1920 / 2 and ys.min() > 1080 / 3, "insets not in the bottom-left"
    # Two insets, stacked: the drawn rows form two separate bands.
    rows = np.nonzero(drawn.any(axis=1))[0]
    assert (np.diff(rows) > 1).sum() == 1, "expected exactly two insets"
    # Nearest (the green one, 5 km) sits in the corner itself, i.e. the lower inset.
    gy = np.nonzero(green(out))[0]
    assert len(gy) and gy.min() > rows[np.nonzero(np.diff(rows) > 1)[0][0]]


def test_edges_and_junk_settings_do_not_crash():
    frame = frame_with_dot(3, 3)
    out = np.zeros((1080, 1920, 3), np.uint8)
    for opts in ({}, {"pip_count": "lots", "pip_zoom": None, "pip_corner": "middle"},
                 {"pip_count": 99, "pip_zoom": 1e9}, {"pip_zoom": float("nan")}):
        draw_pip(out, frame, [aircraft(0.5, 1079.4), aircraft(1919.6, 0.2)], 1.0, 1.0, opts)
    draw_pip(out, frame, [], 1.0, 1.0, {})


def test_settings_are_validated_on_the_way_in():
    import tempfile

    from adsb_camera.app import AdsbCamera
    from adsb_camera.config import load_config

    with tempfile.TemporaryDirectory() as tmp:
        cfg = load_config(None)
        cfg["data_dir"] = tmp
        core = AdsbCamera(cfg)
        core.set_settings({"overlay": {"pip": 1, "pip_count": 9, "pip_zoom": "x",
                                       "pip_corner": "middle"}})
        ov = core.state()["settings"]["overlay"]
        assert ov["pip"] is True and ov["pip_count"] == 4
        assert ov["pip_zoom"] == 4.0 and ov["pip_corner"] == "top-right"
        core.set_settings({"overlay": {"pip_count": 2.6, "pip_zoom": 0, "pip_corner": "bottom-left"}})
        ov = core.state()["settings"]["overlay"]
        assert ov["pip_count"] == 3 and ov["pip_zoom"] == 1.5 and ov["pip_corner"] == "bottom-left"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
