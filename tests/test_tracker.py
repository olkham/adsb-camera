"""The live tracker must not make aircraft jump backwards when a new fix lands.

A fix is stamped with the time we received it, and the delay between the
aircraft being there and us hearing about it varies (0.1-0.9 s here). Dead
reckoning from one fix and snapping to the next then jumps whenever the new
fix was more delayed than the last - backwards along the track. The Kalman
filter with easing should move steadily forward and be no less accurate.

Run:  python -m pytest -q      (or: python tests/test_tracker.py)
"""
import math
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adsb_camera.aircraft import interp_track  # noqa: E402
from adsb_camera.calibration import Site  # noqa: E402
from adsb_camera.geo import destination  # noqa: E402
from adsb_camera.tracker import Tracker, clean_settings  # noqa: E402

SITE = Site(51.4700, -0.4543, 60.0)
FRAME = SITE.frame()
GS, TRK, ALT = 230.0, 70.0, 9000.0
START = destination(SITE.lat, SITE.lon, 200.0, 15000.0)
MEAN_LATENCY = 0.5


def true_pos(s):
    lat, lon = destination(*START, TRK, GS * s)
    return lat, lon, ALT


def make_fixes(seed=1, n=120):
    rng = random.Random(seed)
    fixes = []
    for k in range(n):
        s = float(k)
        lat, lon, alt = true_pos(s)
        fixes.append([s + rng.uniform(0.1, 0.9), lat, lon, alt,
                      GS + rng.gauss(0, 1.0), TRK + rng.gauss(0, 0.3), 0.0])
    return fixes


def fly(settings, video_latency, seed=1):
    """Render at 15 fps; returns (worst backwards step m, RMS error m)."""
    fixes = make_fixes(seed)
    sync = video_latency - MEAN_LATENCY          # what calibration would fit
    tr = Tracker(settings, coast_s=30.0)
    direction = FRAME.enu(*true_pos(1.0)) - FRAME.enu(*true_pos(0.0))
    direction /= np.linalg.norm(direction)
    prev, worst, errs = None, 0.0, []
    for i in range(int(20 * 15), int(100 * 15)):
        wall = i / 15.0
        seen = [f for f in fixes if f[0] <= wall]
        est = tr.estimator(FRAME)("ABC123", seen, wall - sync)
        assert est is not None
        along = float(est @ direction)
        if prev is not None:
            worst = min(worst, along - prev)
        prev = along
        errs.append(np.linalg.norm(est - FRAME.enu(*true_pos(wall - video_latency))))
    return -worst, float(np.sqrt(np.mean(np.square(errs))))


def test_kalman_with_easing_does_not_jump_back():
    for video_latency in (0.0, 2.0):        # dead reckoning ahead / interpolating behind
        raw_back, raw_rms = fly({"mode": "raw", "blend_s": 0}, video_latency)
        kf_back, kf_rms = fly({"mode": "kalman", "blend_s": 1.0}, video_latency)
        print(f"video latency {video_latency}s: raw back {raw_back:.0f} m, rms {raw_rms:.0f} m | "
              f"kalman back {kf_back:.1f} m, rms {kf_rms:.0f} m")
        assert kf_back < 5.0, f"kalman still steps back {kf_back:.1f} m"
        assert kf_rms < raw_rms * 1.1, (kf_rms, raw_rms)
    # The raw estimator does jump back when it dead reckons: the case this is for.
    assert fly({"mode": "raw", "blend_s": 0}, 0.0)[0] > 30.0


def test_easing_alone_smooths_the_raw_estimator():
    back, _ = fly({"mode": "raw", "blend_s": 1.0}, 0.0)
    snap, _ = fly({"mode": "raw", "blend_s": 0}, 0.0)
    assert back < 0.5 * snap, (back, snap)


def test_raw_without_easing_matches_the_calibration_interpolation():
    """Straight lines in ENU vs in lat/lon: the same to well under a metre here."""
    fixes = make_fixes()
    tr = Tracker({"mode": "raw", "blend_s": 0}, coast_s=30.0)
    est = tr.estimator(FRAME)
    for t in np.arange(5.0, 125.0, 0.37):
        a = est("ABC123", fixes, float(t))
        b = FRAME.enu(*interp_track(fixes, float(t), max_extrapolate_s=30.0))
        assert np.linalg.norm(a - b) < 1.0, (t, a, b)


def test_stale_aircraft_disappear():
    fixes = make_fixes(n=10)
    for mode in ("raw", "kalman"):
        est = Tracker({"mode": mode}, coast_s=30.0).estimator(FRAME)
        assert est("ABC123", fixes, fixes[-1][0] + 29.0) is not None
        assert est("ABC123", fixes, fixes[-1][0] + 31.0) is None


def test_settings_change_restarts_cleanly():
    fixes = make_fixes()
    tr = Tracker({"mode": "kalman"}, coast_s=30.0)
    a = tr.estimator(FRAME)("ABC123", fixes, 60.0)
    tr.configure({"mode": "raw", "blend_s": 0})
    b = tr.estimator(FRAME)("ABC123", fixes, 60.0)
    c = FRAME.enu(*interp_track(fixes, 60.0))
    assert np.linalg.norm(b - c) < 1.0, "raw after a switch must not carry kalman state"
    assert np.linalg.norm(a - c) < 150.0


def test_settings_are_validated():
    s = clean_settings({"mode": "bogus", "blend_s": 99, "pos_noise_m": "x",
                        "accel_noise": -1, "use_velocity": 0, "unknown": 1})
    assert s["mode"] == "kalman" and s["blend_s"] == 10.0 and s["pos_noise_m"] == 100.0
    assert s["accel_noise"] == 0.05 and s["use_velocity"] is False and "unknown" not in s
    assert clean_settings({"blend_s": float("nan")})["blend_s"] == 1.0


def test_app_applies_and_saves_tracking_settings():
    import tempfile

    from adsb_camera.app import AdsbCamera
    from adsb_camera.config import load_config

    with tempfile.TemporaryDirectory() as tmp:
        cfg = load_config(None)
        cfg["data_dir"] = tmp
        core = AdsbCamera(cfg)
        core.set_settings({"tracking": {"mode": "raw", "blend_s": 2.5, "pos_noise_m": 1e9}})
        st = core.state()["settings"]["tracking"]
        assert st["mode"] == "raw" and st["blend_s"] == 2.5 and st["pos_noise_m"] == 2000.0
        assert core.tracker.settings["mode"] == "raw"
        again = AdsbCamera(cfg)                     # reloaded from calibration.json
        assert again.tracker.settings["blend_s"] == 2.5


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
