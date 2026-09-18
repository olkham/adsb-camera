"""Camera model: orientation in the local ENU frame + intrinsics + distortion.

Camera axes follow OpenCV: x right, y down, z forward (optical axis).

Two lens models, both with square pixels (fx == fy == f):

* ``pinhole``  – Brown radial distortion (k1, k2, k3). Normal webcams.
* ``fisheye``  – Kannala-Brandt / OpenCV-fisheye (k1..k4), r = f * theta_d.
                 Handles 180°+ lenses (e.g. the Ximea with a fisheye).

Orientation is stored as a rotation matrix (camera -> ENU). Yaw/pitch/roll are
derived for display and slider control only; the solver works on rotation
vectors so a zenith-pointing fisheye has no gimbal lock.

  yaw   : azimuth of the optical axis, degrees clockwise from true north
  pitch : elevation of the optical axis above the horizon, degrees
  roll  : rotation about the optical axis, degrees (positive = clockwise image)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

MODELS = ("pinhole", "fisheye")
N_DIST = {"pinhole": 3, "fisheye": 4}

# Camera axes expressed in ENU for a level camera looking due north.
_R0_C2W = np.array([
    [1.0, 0.0, 0.0],   # cam x (right) -> east
    [0.0, 0.0, 1.0],   # cam z (fwd)   -> north
    [0.0, -1.0, 0.0],  # cam y (down)  -> -up
])


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def ypr_to_c2w(yaw, pitch, roll) -> np.ndarray:
    y, p, r = map(math.radians, (yaw, pitch, roll))
    return _rz(-y) @ _rx(p) @ _R0_C2W @ _rz(r)


def c2w_to_ypr(R: np.ndarray):
    axis = R[:, 2]
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, axis[2]))))
    if abs(axis[2]) > 0.999999:  # straight up/down: fold yaw into roll
        yaw = 0.0
    else:
        yaw = (math.degrees(math.atan2(axis[0], axis[1])) + 360.0) % 360.0
    base = _rz(-math.radians(yaw)) @ _rx(math.radians(pitch)) @ _R0_C2W
    Rr = base.T @ R
    roll = math.degrees(math.atan2(Rr[1, 0], Rr[0, 0]))
    return yaw, pitch, roll


def focal_from_hfov(model: str, hfov_deg: float, width: int) -> float:
    half = math.radians(max(1.0, hfov_deg)) / 2.0
    if model == "pinhole":
        half = min(half, math.radians(89.0))
        return (width / 2.0) / math.tan(half)
    return (width / 2.0) / half


@dataclass
class CameraModel:
    model: str = "pinhole"
    width: int = 1920
    height: int = 1080
    f: float = 1400.0
    cx: float = 960.0
    cy: float = 540.0
    dist: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    R_c2w: np.ndarray = field(default_factory=lambda: _R0_C2W.copy())
    sync_offset_s: float = 0.0
    calibrated: bool = False
    rms_px: float = float("nan")

    # ── construction ──────────────────────────────────────────────────────
    @classmethod
    def default(cls, width=1920, height=1080, model="pinhole", hfov=None,
                yaw=0.0, pitch=10.0, roll=0.0) -> "CameraModel":
        if hfov is None:
            hfov = 180.0 if model == "fisheye" else 70.0
        return cls(model=model, width=int(width), height=int(height),
                   f=focal_from_hfov(model, hfov, width),
                   cx=width / 2.0, cy=height / 2.0,
                   dist=[0.0] * N_DIST[model],
                   R_c2w=ypr_to_c2w(yaw, pitch, roll))

    def copy(self) -> "CameraModel":
        return CameraModel(self.model, self.width, self.height, self.f, self.cx, self.cy,
                           list(self.dist), self.R_c2w.copy(), self.sync_offset_s,
                           self.calibrated, self.rms_px)

    def with_model(self, model: str) -> "CameraModel":
        """Same orientation / FOV, different lens model."""
        if model == self.model:
            return self.copy()
        c = self.copy()
        hfov = self.hfov()
        c.model = model
        c.dist = [0.0] * N_DIST[model]
        c.f = focal_from_hfov(model, min(hfov, 170.0) if model == "pinhole" else hfov, self.width)
        return c

    def scaled_to(self, width: int, height: int) -> "CameraModel":
        """Adapt to a new stream resolution (same sensor, same aspect)."""
        if (width, height) == (self.width, self.height):
            return self
        c = self.copy()
        sx, sy = width / self.width, height / self.height
        c.f *= sx
        c.cx *= sx
        c.cy *= sy
        c.width, c.height = int(width), int(height)
        return c

    # ── orientation helpers ───────────────────────────────────────────────
    @property
    def R_w2c(self) -> np.ndarray:
        return self.R_c2w.T

    def ypr(self):
        return c2w_to_ypr(self.R_c2w)

    def set_ypr(self, yaw, pitch, roll):
        self.R_c2w = ypr_to_c2w(yaw, pitch, roll)

    # ── projection ────────────────────────────────────────────────────────
    def project_cam(self, pc: np.ndarray):
        """Camera-frame points (N,3) -> pixels (N,2), valid mask (N,)."""
        return project(self.model, self.f, self.cx, self.cy, self.dist,
                       np.atleast_2d(pc), self.width, self.height)

    def project_world(self, enu: np.ndarray):
        pc = np.atleast_2d(enu) @ self.R_c2w  # == (R_w2c @ enu.T).T
        return self.project_cam(pc)

    def in_image(self, uv: np.ndarray, margin: float = 0.0) -> np.ndarray:
        return ((uv[:, 0] >= -margin) & (uv[:, 0] < self.width + margin) &
                (uv[:, 1] >= -margin) & (uv[:, 1] < self.height + margin))

    def unproject(self, uv: np.ndarray) -> np.ndarray:
        """Pixels (N,2) -> unit rays in the camera frame (N,3)."""
        uv = np.atleast_2d(np.asarray(uv, dtype=float))
        x = (uv[:, 0] - self.cx) / self.f
        y = (uv[:, 1] - self.cy) / self.f
        rd = np.hypot(x, y)
        k = list(self.dist) + [0.0] * 4
        if self.model == "pinhole":
            r = rd.copy()
            for _ in range(30):  # fixed-point inversion of r*(1+k1r²+k2r⁴+k3r⁶)
                r2 = r * r
                r = rd / (1 + k[0] * r2 + k[1] * r2 * r2 + k[2] * r2 ** 3)
            scale = np.where(rd > 1e-12, r / np.maximum(rd, 1e-12), 1.0)
            rays = np.stack([x * scale, y * scale, np.ones_like(x)], axis=1)
        else:
            th = rd.copy()
            for _ in range(20):  # Newton on theta_d(theta) = rd
                t2 = th * th
                g = th * (1 + k[0] * t2 + k[1] * t2 ** 2 + k[2] * t2 ** 3 + k[3] * t2 ** 4) - rd
                dg = 1 + 3 * k[0] * t2 + 5 * k[1] * t2 ** 2 + 7 * k[2] * t2 ** 3 + 9 * k[3] * t2 ** 4
                th = th - g / np.where(np.abs(dg) < 1e-9, 1e-9, dg)
            s = np.sin(th)
            with np.errstate(invalid="ignore", divide="ignore"):
                ux = np.where(rd > 1e-12, x / rd, 0.0)
                uy = np.where(rd > 1e-12, y / rd, 0.0)
            rays = np.stack([s * ux, s * uy, np.cos(th)], axis=1)
        return rays / np.linalg.norm(rays, axis=1, keepdims=True)

    def pixel_to_world_dir(self, uv) -> np.ndarray:
        return self.unproject(uv) @ self.R_c2w.T

    def hfov(self) -> float:
        return self._span_deg([[0, self.cy], [self.width, self.cy]])

    def vfov(self) -> float:
        return self._span_deg([[self.cx, 0], [self.cx, self.height]])

    def _span_deg(self, pts) -> float:
        r = self.unproject(np.array(pts, dtype=float))
        a = math.degrees(math.acos(max(-1.0, min(1.0, float(r[0] @ np.array([0, 0, 1.0]))))))
        b = math.degrees(math.acos(max(-1.0, min(1.0, float(r[1] @ np.array([0, 0, 1.0]))))))
        return a + b

    # ── persistence ───────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        yaw, pitch, roll = self.ypr()
        return {
            "model": self.model, "width": self.width, "height": self.height,
            "f": self.f, "cx": self.cx, "cy": self.cy, "dist": list(map(float, self.dist)),
            "R_c2w": self.R_c2w.tolist(), "sync_offset_s": self.sync_offset_s,
            "calibrated": self.calibrated,
            "rms_px": None if not math.isfinite(self.rms_px) else self.rms_px,
            "yaw": yaw, "pitch": pitch, "roll": roll,
            "hfov": self.hfov(), "vfov": self.vfov(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CameraModel":
        model = d.get("model", "pinhole")
        dist = list(d.get("dist") or [])
        dist = (dist + [0.0] * N_DIST[model])[: N_DIST[model]]
        R = np.array(d["R_c2w"], dtype=float) if d.get("R_c2w") else ypr_to_c2w(
            d.get("yaw", 0), d.get("pitch", 10), d.get("roll", 0))
        rms = d.get("rms_px")
        return cls(model=model, width=int(d["width"]), height=int(d["height"]),
                   f=float(d["f"]), cx=float(d["cx"]), cy=float(d["cy"]), dist=dist,
                   R_c2w=R, sync_offset_s=float(d.get("sync_offset_s", 0.0)),
                   calibrated=bool(d.get("calibrated", False)),
                   rms_px=float(rms) if rms is not None else float("nan"))


def project(model, f, cx, cy, dist, pc, width, height):
    """Vectorised projection used by both rendering and the solver."""
    X, Y, Z = pc[:, 0], pc[:, 1], pc[:, 2]
    k = list(dist) + [0.0] * 4
    if model == "pinhole":
        valid = Z > 1e-6
        Zs = np.where(valid, Z, 1.0)
        x, y = X / Zs, Y / Zs
        r2 = x * x + y * y
        radial = 1 + k[0] * r2 + k[1] * r2 * r2 + k[2] * r2 ** 3
        # Distortion polynomial must stay monotonic, otherwise far-off points
        # fold back into the image.
        slope = 1 + 3 * k[0] * r2 + 5 * k[1] * r2 * r2 + 7 * k[2] * r2 ** 3
        rmax2 = (1.5 * math.hypot(width, height) / 2.0 / f) ** 2
        valid &= (slope > 0) & (r2 < rmax2) & (radial > 0)
        u = f * x * radial + cx
        v = f * y * radial + cy
    else:
        rho = np.hypot(X, Y)
        th = np.arctan2(rho, Z)
        t2 = th * th
        thd = th * (1 + k[0] * t2 + k[1] * t2 ** 2 + k[2] * t2 ** 3 + k[3] * t2 ** 4)
        slope = 1 + 3 * k[0] * t2 + 5 * k[1] * t2 ** 2 + 7 * k[2] * t2 ** 3 + 9 * k[3] * t2 ** 4
        valid = (slope > 0) & (th < math.radians(135.0))
        with np.errstate(invalid="ignore", divide="ignore"):
            s = np.where(rho > 1e-12, thd / np.maximum(rho, 1e-12), 0.0)
        u = f * X * s + cx
        v = f * Y * s + cy
    return np.stack([u, v], axis=1), valid
