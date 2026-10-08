"""Where is everything in the image right now?

Pure functions from (camera, site, aircraft snapshot, frame time) to a list of
aircraft with pixel positions, plus the az/el reference grid.
"""
from __future__ import annotations

import math

import numpy as np

from .aircraft import ALT, LAT, LON, T, interp_track
from .camera_model import CameraModel
from .geo import az_el_range, az_el_to_enu

# FlightRadar-style altitude colours (same stops as adsb-mapper), RGB.
_ALT_STOPS = [(0, (255, 77, 77)), (10000, (255, 157, 0)), (20000, (255, 224, 0)),
              (30000, (105, 216, 58)), (40000, (24, 179, 255)), (45000, (155, 107, 255))]


def altitude_rgb(ft) -> tuple:
    if not isinstance(ft, (int, float)) or not math.isfinite(ft):
        return (200, 200, 200)
    if ft <= _ALT_STOPS[0][0]:
        return _ALT_STOPS[0][1]
    for (a, ca), (b, cb) in zip(_ALT_STOPS, _ALT_STOPS[1:]):
        if ft <= b:
            w = (ft - a) / (b - a)
            return tuple(int(ca[i] + w * (cb[i] - ca[i])) for i in range(3))
    return _ALT_STOPS[-1][1]


def compute_scene(cam: CameraModel, frame, snapshot: dict, t: float,
                  max_range_km: float, stale_s: float, trail_s: float,
                  with_trails: bool = True, estimate=None) -> list:
    """Aircraft at frame time t (shifted by the camera's sync offset).

    ``estimate(icao, fixes, t_world) -> ENU or None`` replaces the plain
    interpolation (the live view passes its tracker's)."""
    t_world = t - cam.sync_offset_s
    out = []
    for icao, ac in snapshot.items():
        fixes = ac["fixes"]
        if estimate is not None:
            enu = estimate(icao, fixes, t_world)
            if enu is None:
                continue
        else:
            p = interp_track(fixes, t_world, max_extrapolate_s=stale_s)
            if p is None:
                continue
            enu = frame.enu(*p)
        az, el, rng = (float(x) for x in az_el_range(enu))
        if rng > max_range_km * 1000.0:
            continue
        uv, valid = cam.project_world(enu)
        f = ac["fields"]
        alt_ft = f.get("altitude_ft")
        entry = {
            "icao": icao,
            "callsign": ac.get("callsign") or "",
            "alt_ft": alt_ft if isinstance(alt_ft, (int, float)) else None,
            "on_ground": alt_ft == "ground",
            "gs_kt": f.get("ground_speed_kt"),
            "vrate_fpm": f.get("vertical_rate_fpm"),
            "track": f.get("track"),
            "squawk": f.get("squawk"),
            "emergency": f.get("emergency"),
            "az": az, "el": el, "range_km": rng / 1000.0,
            "age_s": max(0.0, t_world - fixes[-1][T]),
            "uv": uv[0].tolist() if valid[0] else None,
            "in_view": bool(valid[0] and cam.in_image(uv)[0]),
            "rgb": altitude_rgb(alt_ft if not isinstance(alt_ft, str) else 0),
            "trail": None,
        }
        if with_trails and entry["uv"] is not None:
            past = [x for x in fixes if t_world - trail_s <= x[T] <= t_world]
            if past:
                arr = np.array([[x[LAT], x[LON], x[ALT]] for x in past])
                pts = np.vstack([frame.enu(arr[:, 0], arr[:, 1], arr[:, 2]), enu[None]])
                tuv, tvalid = cam.project_world(pts)
                entry["trail"] = [q.tolist() if ok else None for q, ok in zip(tuv, tvalid)]
        out.append(entry)
    out.sort(key=lambda a: a["range_km"])
    return out


def _polylines(cam: CameraModel, enu_pts: np.ndarray):
    """Project a sampled curve, splitting where it leaves the valid region or jumps."""
    uv, valid = cam.project_world(enu_pts)
    lines, cur = [], []
    jump = max(cam.width, cam.height) / 4
    for q, ok in zip(uv, valid):
        if ok and cur and np.hypot(*(q - cur[-1])) > jump:
            ok = False
        if ok:
            cur.append(q)
        elif len(cur) > 1:
            lines.append(np.array(cur))
            cur = []
        else:
            cur = []
    if len(cur) > 1:
        lines.append(np.array(cur))
    return lines


def sky_grid(cam: CameraModel) -> dict:
    """Horizon, elevation rings, azimuth spokes and labels in pixels."""
    az = np.arange(0.0, 360.5, 0.5)
    rings = {}
    for el in (0, 10, 20, 30, 45, 60, 75):
        rings[el] = _polylines(cam, az_el_to_enu(az, np.full_like(az, el)))
    spokes = {}
    els = np.arange(-5.0, 88.0, 0.5)
    for a in range(0, 360, 10):
        spokes[a] = _polylines(cam, az_el_to_enu(np.full_like(els, a), els))
    labels = []
    names = {0: "N", 45: "NE", 90: "E", 135: "SE", 180: "S", 225: "SW", 270: "W", 315: "NW"}
    for a in range(0, 360, 15):
        uv, ok = cam.project_world(az_el_to_enu(a, 1.0)[None])
        if ok[0] and cam.in_image(uv)[0]:
            labels.append((uv[0], names.get(a, f"{a}"), a in names))
    for el in (10, 20, 30, 45, 60, 75):
        # Label rings at the azimuth the camera faces.
        yaw = cam.ypr()[0]
        uv, ok = cam.project_world(az_el_to_enu(yaw, el)[None])
        if ok[0] and cam.in_image(uv)[0]:
            labels.append((uv[0], f"{el}deg", False))
    return {"rings": rings, "spokes": spokes, "labels": labels}


def fov_outline(cam: CameraModel, n: int = 24) -> list:
    """Image border as [[az, el], ...] for the sky plot (skips unimaged corners)."""
    w, h = cam.width, cam.height
    s = np.linspace(0, 1, n, endpoint=False)
    border = np.concatenate([
        np.stack([s * w, np.zeros(n)], 1), np.stack([np.full(n, w), s * h], 1),
        np.stack([(1 - s) * w, np.full(n, h)], 1), np.stack([np.zeros(n), (1 - s) * h], 1)])
    rays = cam.unproject(border)
    back, ok = cam.project_cam(rays)
    good = ok & (np.linalg.norm(back - border, axis=1) < 2.0)
    az, el, _ = az_el_range(rays[good] @ cam.R_c2w.T)
    return [[round(float(a), 2), round(float(e), 2)] for a, e in zip(az, el)]
