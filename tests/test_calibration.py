"""Synthetic end-to-end tests: build a 'true' camera, fly fake aircraft past it,
click on them (with pixel noise and a video/ADS-B sync offset), and check the
solver recovers the camera from a naive starting guess.

Run:  python -m pytest -q      (or: python tests/test_calibration.py)
"""
import math
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adsb_camera.aircraft import interp_track  # noqa: E402
from adsb_camera.calibration import Correspondence, Site, SolveOptions, solve  # noqa: E402
from adsb_camera.camera_model import CameraModel, c2w_to_ypr, ypr_to_c2w  # noqa: E402
from adsb_camera.geo import destination  # noqa: E402

SITE = Site(51.4700, -0.4543, 60.0)


def make_track(rng, t0, t1):
    brg = rng.uniform(0, 360)
    dist = rng.uniform(4000, 45000)
    lat, lon = destination(SITE.lat, SITE.lon, brg, dist)
    trk, gs, alt = rng.uniform(0, 360), rng.uniform(110, 240), rng.uniform(900, 11000)
    vr = rng.uniform(-8, 8)
    fixes, t = [], t0
    while t <= t1:
        la, lo = destination(lat, lon, trk, gs * (t - t0))
        fixes.append([t, la, lo, alt + vr * (t - t0), gs, trk, vr])
        t += 2.0
    return fixes


def simulate(true_cam, n_points, sync_true=1.3, noise_px=0.8, seed=1):
    rng = random.Random(seed)
    frame = SITE.frame()
    pts, tries = [], 0
    while len(pts) < n_points and tries < 20000:
        tries += 1
        t_click = 1000.0 + rng.uniform(0, 600)
        fixes = make_track(rng, t_click - 80, t_click + 80)
        p = interp_track(fixes, t_click - sync_true)
        enu = frame.enu(*p)
        uv, valid = true_cam.project_world(enu)
        if not valid[0] or not true_cam.in_image(uv, margin=-20)[0]:
            continue
        u, v = uv[0] + np.random.default_rng(len(pts)).normal(0, noise_px, 2)
        # Only keep fixes that would have been received by click time + a bit.
        kept = [f for f in fixes if f[0] <= t_click + 4]
        pts.append(Correspondence(u=float(u), v=float(v), img_w=true_cam.width,
                                  img_h=true_cam.height, t=t_click, icao=f"T{len(pts):05d}",
                                  fixes=kept))
    return pts


def ang_err(a, b):
    R = a.R_c2w.T @ b.R_c2w
    return math.degrees(math.acos(max(-1, min(1, (np.trace(R) - 1) / 2))))


def overlay_err(cam, true, margin=0.1):
    """Max pixel disagreement between the fitted and true camera over the
    image: what actually matters for the overlay (rotation and principal
    point trade off against each other, so compare their joint effect)."""
    xs = np.linspace(true.width * margin, true.width * (1 - margin), 9)
    ys = np.linspace(true.height * margin, true.height * (1 - margin), 7)
    uv = np.array([[x, y] for x in xs for y in ys])
    if true.model == "fisheye":  # stay inside the image circle
        r = np.hypot(uv[:, 0] - true.cx, uv[:, 1] - true.cy)
        uv = uv[r < 0.8 * min(true.width, true.height) / 2]
    dirs = true.pixel_to_world_dir(uv)
    est, valid = cam.project_world(dirs)
    assert valid.all()
    return float(np.max(np.linalg.norm(est - uv, axis=1)))


def test_ypr_roundtrip():
    for y, p, r in [(0, 0, 0), (123, 20, -7), (300, -30, 45), (45, 89, 0)]:
        yy, pp, rr = c2w_to_ypr(ypr_to_c2w(y, p, r))
        R1, R2 = ypr_to_c2w(y, p, r), ypr_to_c2w(yy, pp, rr)
        assert np.allclose(R1, R2, atol=1e-6)


def test_projection_unprojection_roundtrip():
    for cam in (CameraModel.default(1920, 1080, "pinhole", 70),
                CameraModel.default(2048, 2048, "fisheye", 190)):
        cam.dist = [-0.1, 0.02, 0.0] if cam.model == "pinhole" else [0.02, -0.01, 0.0, 0.0]
        # Points inside the lens image circle (fisheye corners are unimaged).
        uv = np.array([[cam.cx - 600, cam.cy - 300], [cam.cx, cam.cy],
                       [cam.cx + 650, cam.cy + 420]], float)
        back, valid = cam.project_cam(cam.unproject(uv))
        assert valid.all() and np.allclose(back, uv, atol=1e-3)


def test_recover_webcam():
    true = CameraModel.default(1920, 1080, "pinhole", 64, yaw=231, pitch=11, roll=1.5)
    true.dist, true.cx, true.cy = [-0.12, 0.04, 0.0], 975.0, 530.0
    pts = simulate(true, 24)
    guess = CameraModel.default(1920, 1080, "pinhole")  # facing north, 70° — way off
    res = solve(guess, pts, SITE, SolveOptions(model="auto"))
    assert res["ok"], res["message"]
    cam = res["camera"]
    print(res["message"], "| sync", round(cam.sync_offset_s, 2), "| rot err",
          round(ang_err(cam, true), 3), "| hfov", round(cam.hfov(), 2), "vs", round(true.hfov(), 2),
          "| overlay err", round(overlay_err(cam, true), 2), "px")
    assert res["model"] == "pinhole"
    assert res["rms_px"] < 3.0
    assert overlay_err(cam, true) < 4.0
    assert abs(cam.hfov() - true.hfov()) < 1.0
    assert abs(cam.sync_offset_s - 1.3) < 0.5


def test_two_clicks_gets_close():
    true = CameraModel.default(1920, 1080, "pinhole", 70, yaw=95, pitch=15, roll=0)
    pts = simulate(true, 2, sync_true=0.0, seed=7)
    res = solve(CameraModel.default(1920, 1080, "pinhole"), pts, SITE, SolveOptions())
    assert res["ok"], res["message"]
    # Two clicks: rotation + fixed default FOV only, so centre is decent.
    c = res["camera"]
    centre = true.pixel_to_world_dir(np.array([[true.cx, true.cy]]))
    uv, _ = c.project_world(centre)
    assert np.linalg.norm(uv[0] - [true.cx, true.cy]) < 40


def test_recover_fisheye_zenith():
    true = CameraModel.default(2048, 2048, "fisheye", 185, yaw=40, pitch=88, roll=0)
    true.dist = [0.03, -0.01, 0.0, 0.0]
    pts = simulate(true, 30, seed=3)
    res = solve(CameraModel.default(2048, 2048, "pinhole"), pts, SITE, SolveOptions(model="auto"))
    assert res["ok"], res["message"]
    cam = res["camera"]
    print(res["message"], "| overlay err", round(overlay_err(cam, true), 2), "px")
    assert res["model"] == "fisheye"
    assert res["rms_px"] < 3.0
    assert overlay_err(cam, true) < 5.0


def test_outlier_is_flagged():
    true = CameraModel.default(1920, 1080, "pinhole", 60, yaw=180, pitch=10)
    pts = simulate(true, 16, seed=11)
    pts[3].u += 250  # mis-click / wrong aircraft
    res = solve(CameraModel.default(1920, 1080, "pinhole"), pts, SITE, SolveOptions())
    assert res["ok"]
    assert pts[3].id in res["outliers"]
    assert res["rms_px"] < 3.0
    assert overlay_err(res["camera"], true) < 5.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
