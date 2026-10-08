"""Locked parameters must come out of a solve exactly as they went in, while the
free ones are still fitted - e.g. trust the camera's lat/lon but let the solver
correct its height.

Run:  python -m pytest -q      (or: python tests/test_locks.py)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adsb_camera.calibration import (DEFAULT_LOCKS, POSITION, Site, SolveOptions,  # noqa: E402
                                     normalise_solve_opts, solve)
from adsb_camera.camera_model import CameraModel  # noqa: E402
from test_calibration import SITE, overlay_err, simulate  # noqa: E402

TRUE = CameraModel.default(1920, 1080, "pinhole", 64, yaw=231, pitch=11, roll=1.5)


def locks(*names):
    return frozenset(names) | frozenset(POSITION)


def test_locked_yaw_stays_put_and_the_rest_is_fitted():
    pts = simulate(TRUE, 20)
    guess = CameraModel.default(1920, 1080, "pinhole", 70, yaw=231, pitch=0, roll=0)
    res = solve(guess, pts, SITE, SolveOptions(locks=locks("yaw")))
    assert res["ok"], res["message"]
    y, p, r = res["camera"].ypr()
    assert abs(y - 231) < 1e-6, f"yaw moved to {y}"
    assert abs(p - 11) < 0.5 and abs(r - 1.5) < 0.5, (p, r)
    assert "rot" not in res["free"] and "pitch" in res["free"] and "yaw" not in res["free"]


def test_wrong_locked_value_is_kept_even_though_it_fits_worse():
    pts = simulate(TRUE, 12)
    guess = CameraModel.default(1920, 1080, "pinhole", 64, yaw=229, pitch=11, roll=1.5)
    res = solve(guess, pts, SITE, SolveOptions(locks=locks("yaw")))
    assert abs(res["camera"].ypr()[0] - 229) < 1e-6


def test_locked_lens_and_sync_stay_put():
    pts = simulate(TRUE, 24)
    guess = CameraModel.default(1920, 1080, "pinhole", 60, yaw=225, pitch=8)
    guess.cx, guess.cy, guess.dist, guess.sync_offset_s = 900.0, 560.0, [-0.05, 0.0, 0.0], 0.7
    res = solve(guess, pts, SITE, SolveOptions(model="auto", locks=locks("f", "cxcy", "dist", "sync")))
    assert res["ok"], res["message"]
    c = res["camera"]
    assert (c.f, c.cx, c.cy, c.dist, c.sync_offset_s) == \
        (guess.f, guess.cx, guess.cy, guess.dist, guess.sync_offset_s)
    assert res["free"] == ["rot"]


def test_height_only_is_refined_with_lat_lon_locked():
    """The camera is 150 m lower than we were told; its lat/lon are right."""
    pts = simulate(TRUE, 30)
    told = Site(SITE.lat, SITE.lon, SITE.alt_m + 150.0, SITE.geoid_m)
    start = CameraModel.default(1920, 1080, "pinhole", 70, yaw=225, pitch=5)
    base = solve(start, pts, told, SolveOptions())
    res = solve(start, pts, told, SolveOptions(locks=frozenset({"lat", "lon"})))
    assert res["ok"], res["message"]
    assert "alt" in res["free"] and "lat" not in res["free"] and "lon" not in res["free"]
    site = res["site"]
    assert site is not None
    print(f"height: told {told.alt_m:.0f}, true {SITE.alt_m:.0f}, fitted {site.alt_m:.0f} m; "
          f"RMS {base['rms_px']:.1f} -> {res['rms_px']:.1f} px")
    assert (site.lat, site.lon) == (told.lat, told.lon), "locked lat/lon moved"
    assert abs(site.alt_m - SITE.alt_m) < 50.0
    assert res["rms_px"] < base["rms_px"]
    assert overlay_err(res["camera"], TRUE) < 5.0


def test_everything_locked_is_refused_not_crashed():
    pts = simulate(TRUE, 6)
    res = solve(CameraModel.default(1920, 1080), pts, SITE,
                SolveOptions(locks=frozenset(DEFAULT_LOCKS)))
    assert not res["ok"] and "locked" in res["message"]


def test_older_settings_are_upgraded():
    d = normalise_solve_opts({"model": "auto", "sync": "off", "lock_intrinsics": True,
                              "solve_position": True, "auto_solve": True})
    assert d["sync"] == "auto" and "lock_intrinsics" not in d and "solve_position" not in d
    lk = d["locks"]
    assert lk["f"] and lk["cxcy"] and lk["dist"] and lk["sync"]
    assert not (lk["lat"] or lk["lon"] or lk["alt"] or lk["yaw"])
    fresh = normalise_solve_opts({})["locks"]
    assert fresh == DEFAULT_LOCKS and fresh["lat"] and not fresh["yaw"]


def test_app_toggles_locks():
    import tempfile

    from adsb_camera.app import AdsbCamera
    from adsb_camera.config import load_config

    with tempfile.TemporaryDirectory() as tmp:
        cfg = load_config(None)
        cfg["data_dir"] = tmp
        core = AdsbCamera(cfg)
        core.set_settings({"solve": {"locks": {"alt": False, "yaw": True, "bogus": True}}})
        lk = core.state()["settings"]["solve"]["locks"]
        assert lk["yaw"] and not lk["alt"] and lk["lat"] and "bogus" not in lk
        assert AdsbCamera(cfg).solve_opts["locks"] == lk, "locks must survive a restart"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
