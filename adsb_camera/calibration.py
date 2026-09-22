"""Self-calibration from aircraft (and optional landmark) sightings.

Each correspondence says "this pixel, in the frame received at time t, is
aircraft X" (or "is the landmark at lat/lon/alt"). With the camera position
known, every correspondence is a world *direction*; the solver finds the
camera orientation, focal length, principal point, lens distortion, the
video/ADS-B sync offset and (optionally) a correction to the camera position
that best reproject those directions onto the clicked pixels.

Parameters are unlocked progressively as points accumulate, with weak priors
on everything except rotation, so a two-click start gives a usable estimate
and twenty well-spread clicks give a full lens model.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .aircraft import interp_track
from .jsonfile import sweep_temp, write_json
from .camera_model import MODELS, N_DIST, CameraModel, focal_from_hfov, project
from .geo import LocalFrame

log = logging.getLogger("adsb_camera.calibration")

FOV_GUESSES = {"pinhole": (30.0, 50.0, 70.0, 90.0, 115.0),
               "fisheye": (120.0, 160.0, 190.0, 220.0)}
TRACK_WINDOW_S = 90.0


@dataclass
class Site:
    """Camera location. alt_m is metres above mean sea level (what a phone or
    map shows); geoid_m converts MSL to WGS-84 ellipsoid height, which is what
    ADS-B geometric altitude uses (~+47 m around London)."""
    lat: Optional[float] = None
    lon: Optional[float] = None
    alt_m: float = 0.0
    geoid_m: float = 0.0

    @property
    def valid(self) -> bool:
        return self.lat is not None and self.lon is not None

    @property
    def ellipsoid_alt(self) -> float:
        return self.alt_m + self.geoid_m

    def frame(self) -> LocalFrame:
        return LocalFrame(self.lat, self.lon, self.ellipsoid_alt)


@dataclass
class Correspondence:
    u: float
    v: float
    img_w: int
    img_h: int
    t: float
    kind: str = "aircraft"               # "aircraft" | "landmark"
    icao: str = ""
    label: str = ""
    fixes: list = field(default_factory=list)
    lat: Optional[float] = None
    lon: Optional[float] = None
    alt_m: Optional[float] = None
    source: str = "manual"               # "manual" | "auto"
    enabled: bool = True
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    created: float = field(default_factory=time.time)
    residual_px: Optional[float] = None

    def pixel_for(self, w: int, h: int):
        return self.u * w / self.img_w, self.v * h / self.img_h

    def merge_fixes(self, new_fixes: list):
        """Add fixes that arrived after the click (turns extrapolation into
        interpolation, which is far more accurate)."""
        if self.kind != "aircraft" or not new_fixes:
            return
        lo, hi = self.t - TRACK_WINDOW_S, self.t + TRACK_WINDOW_S
        known = {round(f[0], 3) for f in self.fixes}
        added = [f for f in new_fixes if lo <= f[0] <= hi and round(f[0], 3) not in known]
        if added:
            self.fixes = sorted(self.fixes + added, key=lambda f: f[0])

    def summary(self, w: int, h: int) -> dict:
        u, v = self.pixel_for(w, h)
        return {"id": self.id, "kind": self.kind, "icao": self.icao, "label": self.label,
                "x": u / w, "y": v / h, "t": self.t, "source": self.source,
                "enabled": self.enabled, "residual_px": self.residual_px}


def trim_fixes(fixes: list, t: float) -> list:
    return [list(f) for f in fixes if t - TRACK_WINDOW_S <= f[0] <= t + TRACK_WINDOW_S]


# ── persistence ────────────────────────────────────────────────────────────
class CalibrationStore:
    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        sweep_temp(os.path.dirname(self.path))

    def load(self) -> dict:
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                d = json.load(fh)
        except (OSError, ValueError) as e:
            log.error("Could not read %s: %s", self.path, e)
            return {}
        out = {}
        if d.get("site"):
            out["site"] = Site(**d["site"])
        if d.get("camera"):
            try:
                out["camera"] = CameraModel.from_dict(d["camera"])
            except (KeyError, ValueError, TypeError) as e:
                log.error("Ignoring bad camera block: %s", e)
        known = set(Correspondence.__dataclass_fields__)
        out["points"] = [Correspondence(**{k: v for k, v in p.items() if k in known})
                         for p in d.get("points", [])]
        out["settings"] = d.get("settings", {})
        return out

    def save(self, site: Site, camera: CameraModel, points: list, settings: dict):
        d = {"version": 1, "saved": time.time(), "site": asdict(site),
             "camera": camera.to_dict(), "points": [asdict(p) for p in points],
             "settings": settings}
        write_json(self.path, d)


# ── geometry of observations ───────────────────────────────────────────────
@dataclass
class _Obs:
    ids: list
    uv: np.ndarray        # (N,2) observed pixels at camera resolution
    P0: np.ndarray        # (N,3) ENU position at reference sync offset
    V: np.ndarray         # (N,3) ENU velocity (m/s); zero for landmarks
    tau0: float
    weight: np.ndarray    # (N,)
    moving: np.ndarray    # (N,) bool


def _build_obs(points, site: Site, cam: CameraModel, tau0: float) -> _Obs:
    frame = site.frame()
    ids, uv, P0, V, w, moving = [], [], [], [], [], []
    for p in points:
        if not p.enabled:
            continue
        if p.kind == "landmark":
            if p.lat is None or p.lon is None:
                continue
            pos = frame.enu(p.lat, p.lon, (p.alt_m or 0.0) + site.geoid_m)  # MSL -> ellipsoid
            vel = np.zeros(3)
        else:
            a = interp_track(p.fixes, p.t - tau0, max_extrapolate_s=30.0)
            b1 = interp_track(p.fixes, p.t - tau0 - 0.5, max_extrapolate_s=30.5)
            b2 = interp_track(p.fixes, p.t - tau0 + 0.5, max_extrapolate_s=30.5)
            if a is None:
                continue
            pos = frame.enu(*a)
            vel = (frame.enu(*b2) - frame.enu(*b1)) if (b1 and b2) else np.zeros(3)
        ids.append(p.id)
        uv.append(p.pixel_for(cam.width, cam.height))
        P0.append(pos)
        V.append(vel)
        w.append(0.5 if p.source == "auto" else 1.0)
        moving.append(float(np.linalg.norm(vel)) > 20.0)
    return _Obs(ids, np.array(uv, float).reshape(-1, 2), np.array(P0, float).reshape(-1, 3),
                np.array(V, float).reshape(-1, 3), tau0, np.array(w, float), np.array(moving, bool))


def _kabsch(world_dirs: np.ndarray, cam_rays: np.ndarray) -> np.ndarray:
    """Rotation R (world->cam) minimising sum |R d - r|^2."""
    H = world_dirs.T @ cam_rays
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


# ── solver ─────────────────────────────────────────────────────────────────
@dataclass
class SolveOptions:
    model: str = "auto"            # auto | pinhole | fisheye
    sync: str = "auto"             # auto | on | off
    solve_position: bool = False
    lock_intrinsics: bool = False  # rotation (+sync) only: re-aim a known camera
    max_nfev: int = 400


def choose_free(n: int, n_moving: int, model: str, opt: SolveOptions) -> list:
    free = ["rot"]
    if opt.lock_intrinsics:
        if opt.sync == "on" or (opt.sync == "auto" and n_moving >= 6):
            free.append("sync")
        return free
    if n >= 3:
        free.append("f")
    if n >= 6:
        free.append("k1")
    if n >= 8:
        free.append("cxcy")
    if n >= 10:
        free.append("k2")
    if model == "fisheye" and n >= 14:
        free.append("k3")
    if opt.sync == "on" or (opt.sync == "auto" and n_moving >= 8):
        free.append("sync")
    if opt.solve_position and n >= 15:
        free.append("pos")
    return free


class _Problem:
    """Packs/unpacks the parameter vector for a given set of free params."""

    def __init__(self, cam: CameraModel, obs: _Obs, free: list, prior_sync: float):
        self.cam, self.obs, self.free = cam, obs, free
        self.R_base = cam.R_c2w.copy()
        self.nk = sum(1 for k in ("k1", "k2", "k3") if k in free)
        self.prior_sync = prior_sync
        self.penalty = float(max(cam.width, cam.height))

    def x0(self) -> np.ndarray:
        c = self.cam
        x = [0.0, 0.0, 0.0]
        if "f" in self.free:
            x.append(math.log(c.f))
        if "cxcy" in self.free:
            x += [c.cx, c.cy]
        x += list(c.dist[: self.nk])
        if "sync" in self.free:
            x.append(c.sync_offset_s)
        if "pos" in self.free:
            x += [0.0, 0.0, 0.0]
        return np.array(x, float)

    def unpack(self, x):
        c = self.cam
        i = 3
        R = Rotation.from_rotvec(x[:3]).as_matrix() @ self.R_base
        f = c.f
        if "f" in self.free:
            f = math.exp(x[i]); i += 1
        cx, cy = c.cx, c.cy
        if "cxcy" in self.free:
            cx, cy = x[i], x[i + 1]; i += 2
        dist = list(c.dist)
        for j in range(self.nk):
            dist[j] = x[i]; i += 1
        tau = c.sync_offset_s
        if "sync" in self.free:
            tau = x[i]; i += 1
        pos = np.zeros(3)
        if "pos" in self.free:
            pos = np.array(x[i:i + 3]); i += 3
        return R, f, cx, cy, dist, tau, pos

    def residuals(self, x):
        R, f, cx, cy, dist, tau, pos = self.unpack(x)
        o, c = self.obs, self.cam
        P = o.P0 - (tau - o.tau0) * o.V - pos
        d = P / np.linalg.norm(P, axis=1, keepdims=True)
        uv, valid = project(c.model, f, cx, cy, dist, d @ R, c.width, c.height)
        r = (uv - o.uv) * o.weight[:, None]
        r[~valid] = self.penalty
        res = [r.ravel()]
        # Weak priors (pixel-equivalent units) keep under-determined params sane.
        if "cxcy" in self.free:
            res.append(0.02 * np.array([cx - c.width / 2, cy - c.height / 2]))
        if self.nk:
            res.append(40.0 * np.array(dist[: self.nk]))
        if "sync" in self.free:
            res.append(np.array([1.0 * (tau - self.prior_sync)]))
        if "pos" in self.free:
            res.append(pos / 100.0)
        return np.concatenate(res)

    def apply(self, x) -> CameraModel:
        R, f, cx, cy, dist, tau, pos = self.unpack(x)
        out = self.cam.copy()
        out.R_c2w, out.f, out.cx, out.cy, out.dist, out.sync_offset_s = R, f, cx, cy, dist, tau
        return out, pos


def _fit(cam, obs, free, prior_sync, f_scale, max_nfev):
    prob = _Problem(cam, obs, free, prior_sync)
    x0 = prob.x0()
    bounds_lo = np.full_like(x0, -np.inf)
    bounds_hi = np.full_like(x0, np.inf)
    i = 3
    if "f" in free:
        bounds_lo[i], bounds_hi[i] = math.log(cam.width * 0.05), math.log(cam.width * 20)
        i += 1
    if "cxcy" in free:
        bounds_lo[i:i + 2] = [cam.width * 0.2, cam.height * 0.2]
        bounds_hi[i:i + 2] = [cam.width * 0.8, cam.height * 0.8]
        i += 2
    bounds_lo[i:i + prob.nk], bounds_hi[i:i + prob.nk] = -1.5, 1.5
    i += prob.nk
    if "sync" in free:
        bounds_lo[i], bounds_hi[i] = -8.0, 15.0
        i += 1
    if "pos" in free:
        bounds_lo[i:i + 3], bounds_hi[i:i + 3] = -3000.0, 3000.0
    x0 = np.clip(x0, bounds_lo + 1e-9, bounds_hi - 1e-9)
    sol = least_squares(prob.residuals, x0, bounds=(bounds_lo, bounds_hi), loss="soft_l1",
                        f_scale=f_scale, x_scale="jac", max_nfev=max_nfev)
    new_cam, pos = prob.apply(sol.x)
    return new_cam, pos, sol


def reprojection(cam: CameraModel, obs: _Obs, pos=None) -> np.ndarray:
    pos = np.zeros(3) if pos is None else pos
    P = obs.P0 - (cam.sync_offset_s - obs.tau0) * obs.V - pos
    uv, valid = cam.project_world(P)
    err = np.linalg.norm(uv - obs.uv, axis=1)
    err[~valid] = np.inf
    return err


def evaluate(cam: CameraModel, points: list, site: Site) -> dict:
    """Residual per point for the current camera (no fitting)."""
    if not site.valid:
        return {}
    obs = _build_obs(points, site, cam, cam.sync_offset_s)
    if not obs.ids:
        return {}
    err = reprojection(cam, obs)
    return {i: (float(e) if math.isfinite(e) else None) for i, e in zip(obs.ids, err)}


def predict(cam: CameraModel, points: list, site: Site) -> dict:
    """Predicted pixel (at camera resolution) for each enabled point."""
    if not site.valid:
        return {}
    obs = _build_obs(points, site, cam, cam.sync_offset_s)
    if not obs.ids:
        return {}
    uv, valid = cam.project_world(obs.P0)
    return {i: (q.tolist() if ok else None) for i, q, ok in zip(obs.ids, uv, valid)}


def _rms(err: np.ndarray) -> float:
    e = err[np.isfinite(err)]
    return float(np.sqrt(np.mean(e ** 2))) if len(e) else float("inf")


def _robust_rms(err: np.ndarray) -> float:
    e = np.sort(err[np.isfinite(err)])
    return _rms(e[: max(2, int(len(e) * 0.8))]) if len(e) else float("inf")


def solve(cam: CameraModel, points: list, site: Site, opt: SolveOptions) -> dict:
    """Fit the camera to the enabled correspondences. Returns a result dict
    with the new camera under "camera" (None on failure)."""
    if not site.valid:
        return {"ok": False, "message": "Set the camera location first."}
    tau0 = cam.sync_offset_s
    obs = _build_obs(points, site, cam, tau0)
    n = len(obs.ids)
    if n < 2:
        return {"ok": False, "message": f"Need at least 2 usable points (have {n})."}
    if opt.lock_intrinsics:
        models = [cam.model]
    elif opt.model == "auto":
        # Too few points to tell lens models apart: keep the current one.
        models = list(MODELS) if n >= 8 else [cam.model]
    else:
        models = [opt.model]
    n_moving = int(obs.moving.sum())
    t0 = time.time()

    results = []
    for model in models:
        free = choose_free(n, n_moving, model, opt)
        base = cam if (cam.model == model) else cam.with_model(model)
        cands = [base.copy()]
        if not opt.lock_intrinsics:
            dirs = obs.P0 / np.linalg.norm(obs.P0, axis=1, keepdims=True)
            for hfov in FOV_GUESSES[model]:
                c = CameraModel.default(cam.width, cam.height, model, hfov)
                c.sync_offset_s = tau0
                c.R_c2w = _kabsch(dirs, c.unproject(obs.uv)).T
                cands.append(c)
        # Coarse: rotation (+f) only, loose robust scale, from every candidate.
        coarse_free = ["rot"] + (["f"] if "f" in free else [])
        best = None
        for c in cands:
            try:
                fc, _, sol = _fit(c, obs, coarse_free, tau0, 0.02 * cam.width, 150)
            except (ValueError, np.linalg.LinAlgError):
                continue
            if best is None or sol.cost < best[1]:
                best = (fc, sol.cost)
        if best is None:
            continue
        # Fine: everything the point count supports, tight robust scale.
        fc, pos, sol = _fit(best[0], obs, free, tau0, 0.02 * cam.width, opt.max_nfev)
        fc, pos, sol = _fit(fc, obs, free, tau0, 3.0, opt.max_nfev)
        err = reprojection(fc, obs, pos)
        results.append((model, fc, pos, err, free))

    if not results:
        return {"ok": False, "message": "Solver failed to converge."}
    if len(results) == 2:
        # A narrow lens fits both models about equally, so only switch to
        # fisheye when the view is genuinely wide or fisheye clearly wins.
        pin, fish = results
        wide = pin[1].hfov() > 110.0 or not math.isfinite(_robust_rms(pin[3]))
        pick = fish if (wide or _robust_rms(fish[3]) < 0.7 * _robust_rms(pin[3])) else pin
    else:
        pick = results[0]
    model, new_cam, pos, err, free = pick
    med = float(np.median(err[np.isfinite(err)])) if np.isfinite(err).any() else float("inf")
    bad = ~np.isfinite(err)
    if n >= 6:  # with fewer points the fit absorbs errors; flagging would be noise
        bad |= err > max(4 * med, 12.0)
    outliers = [i for i, b in zip(obs.ids, bad) if b]
    rms = _rms(err[~bad]) if (~bad).sum() >= 2 else _rms(err)   # RMS of inliers
    new_cam.calibrated = math.isfinite(rms)
    new_cam.rms_px = rms
    new_site = None
    if "pos" in free and np.linalg.norm(pos) > 0.5:
        new_site = _shift_site(site, pos)
    msg = (f"{model}, {n} points, RMS {rms:.1f}px, fitted: {', '.join(free)}"
           + (f"; {len(outliers)} likely wrong" if outliers else ""))
    log.info("Solve: %s (%.2fs)", msg, time.time() - t0)
    return {"ok": True, "message": msg, "camera": new_cam, "site": new_site,
            "rms_px": rms, "model": model, "free": free, "outliers": outliers,
            "residuals": {i: (float(e) if np.isfinite(e) else None) for i, e in zip(obs.ids, err)},
            "alternatives": {m: _robust_rms(e) for m, _, _, e, _ in results}}


def _shift_site(site: Site, enu_offset: np.ndarray) -> Site:
    """Move the site by an ENU offset (small-offset approximation)."""
    e, n, u = map(float, enu_offset)
    lat = site.lat + math.degrees(n / 6378137.0)
    lon = site.lon + math.degrees(e / (6378137.0 * math.cos(math.radians(site.lat))))
    return Site(lat, lon, site.alt_m + u, site.geoid_m)
