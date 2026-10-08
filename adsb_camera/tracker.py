"""Live position estimates for the overlay.

ADS-B positions arrive about once a second, each stamped with the time *we*
received it. The pipeline's latency varies (readsb batches, adsb2mqtt publishes
on a timer, MQTT queues), so a fix's timestamp is late by a different amount
each time. Dead reckoning from one fix and then snapping to the next therefore
jumps - backwards whenever the new fix was more delayed than the last.

Two estimators, chosen in the Tracking panel:

* ``raw``     straight line between the fixes either side of the frame time,
              dead reckoning past the last one. What calibration uses.
* ``kalman``  a constant-velocity Kalman filter per aircraft in the camera's
              ENU frame, fed each fix's position and (optionally) its reported
              speed, heading and climb rate. Timing jitter becomes position
              noise that the filter averages out against the velocity, which
              ADS-B reports far more steadily.

Whichever is used, a change in the estimate when a new fix lands can be eased
in over ``blend_s`` seconds instead of snapping (the offset decays
exponentially, reaching ~95% after ``blend_s``).

Only the live view goes through here. Frozen and saved frames interpolate
between the fixes either side, which is the best estimate once they're known.
"""
from __future__ import annotations

import math
from bisect import bisect_right
from typing import Callable, Optional

import numpy as np

from .aircraft import ALT, GS, LAT, LON, T, TRK, VR
from .geo import destination

MODES = ("kalman", "raw")
DEFAULTS = {
    "mode": "kalman",
    "blend_s": 1.0,         # ease corrections in over this long; 0 = snap
    "pos_noise_m": 100.0,   # how far a fix may be from the truth (mostly timing jitter x speed)
    "accel_noise": 5.0,     # how hard aircraft may manoeuvre, m/s^2
    "vel_noise_ms": 5.0,    # reported speed / heading / climb rate error, m/s
    "use_velocity": True,
}
LIMITS = {"blend_s": (0.0, 10.0), "pos_noise_m": (1.0, 2000.0),
          "accel_noise": (0.05, 50.0), "vel_noise_ms": (0.1, 100.0)}

ALT_NOISE_M = 15.0      # altitude is reported in 25 ft steps and hardly jitters
V0_STD_MS = 200.0       # velocity uncertainty for a new track with no reported speed
SNAP_M = 3000.0         # a correction bigger than this is a new track, not a wobble
BLEND_GAP_S = 5.0       # frames further apart than this aren't blended across
FORGET_S = 180.0        # drop an aircraft's filter this long after it was last asked for
HISTORY = 256           # filtered states kept per aircraft, for frame times behind the newest fix


def clean_settings(d: Optional[dict], base: Optional[dict] = None) -> dict:
    """``base`` updated with the valid entries of ``d``; junk is ignored, numbers clamped."""
    out = dict(base or DEFAULTS)
    for k, v in (d or {}).items():
        if k == "mode":
            if v in MODES:
                out[k] = v
        elif k == "use_velocity":
            out[k] = bool(v)
        elif k in LIMITS:
            try:
                x = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(x):
                lo, hi = LIMITS[k]
                out[k] = min(hi, max(lo, x))
    return out


class Segment:
    """Straight-line motion p(t) = p0 + v (t - t0) in ENU. ``key`` changes
    whenever the estimate is rebuilt from different data."""
    __slots__ = ("key", "t0", "p0", "v")

    def __init__(self, key, t0: float, p0: np.ndarray, v: np.ndarray):
        self.key, self.t0, self.p0, self.v = key, t0, p0, v

    def at(self, t: float) -> np.ndarray:
        return self.p0 + self.v * (t - self.t0)


def _fix_enu(frame, f) -> np.ndarray:
    return frame.enu(f[LAT], f[LON], f[ALT])


def reported_velocity(frame, f, p: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """ENU velocity (m/s) from a fix's ground speed / track / vertical rate, or None."""
    gs, trk, vr = f[GS], f[TRK], f[VR]
    if not (math.isfinite(gs) and math.isfinite(trk)):
        return None
    p = _fix_enu(frame, f) if p is None else p
    lat, lon = destination(f[LAT], f[LON], trk, gs)       # one second on
    return frame.enu(lat, lon, f[ALT] + (vr if math.isfinite(vr) else 0.0)) - p


def raw_segment(fixes: list, t: float, frame, coast_s: float) -> Optional[Segment]:
    """The interpolate / dead-reckon estimate of ``interp_track``, as a line in ENU."""
    if not fixes:
        return None
    times = [f[T] for f in fixes]
    i = bisect_right(times, t)
    if 0 < i < len(fixes):
        a, b = fixes[i - 1], fixes[i]
        pa, pb = _fix_enu(frame, a), _fix_enu(frame, b)
        span = b[T] - a[T]
        v = (pb - pa) / span if span > 0 else np.zeros(3)
        return Segment(("between", a[T], b[T]), a[T], pa, v)
    ref = fixes[-1] if i == len(fixes) else fixes[0]
    if abs(t - ref[T]) > coast_s:
        return None
    p = _fix_enu(frame, ref)
    v = reported_velocity(frame, ref, p)
    if v is None:
        v = np.zeros(3)
        if len(fixes) >= 2:
            a, b = (fixes[-2], fixes[-1]) if i == len(fixes) else (fixes[0], fixes[1])
            span = b[T] - a[T]
            if span > 0.2:
                v = (_fix_enu(frame, b) - _fix_enu(frame, a)) / span
    return Segment(("from", ref[T]), ref[T], p, v)


class _Kalman:
    """Constant velocity, one independent [position, velocity] filter per ENU
    axis (white-noise acceleration). Axes are kept as length-3 arrays, so every
    step is a handful of vector operations."""
    __slots__ = ("x", "v", "p00", "p01", "p11", "t")

    def __init__(self, t: float, p: np.ndarray, v: Optional[np.ndarray], r_pos, r_vel):
        self.t = t
        self.x = p.copy()
        self.v = np.zeros(3) if v is None else v.copy()
        self.p00 = r_pos.copy()
        self.p01 = np.zeros(3)
        self.p11 = np.full(3, V0_STD_MS ** 2) if v is None else r_vel.copy()

    def predict(self, t: float, q: float):
        dt = t - self.t
        if dt <= 0:
            return
        self.x = self.x + dt * self.v
        self.p00 = self.p00 + 2 * dt * self.p01 + dt * dt * self.p11 + q * dt ** 3 / 3
        self.p01 = self.p01 + dt * self.p11 + q * dt * dt / 2
        self.p11 = self.p11 + q * dt
        self.t = t

    def update_pos(self, z: np.ndarray, r: np.ndarray):
        s = self.p00 + r
        k0, k1 = self.p00 / s, self.p01 / s
        y = z - self.x
        self.x = self.x + k0 * y
        self.v = self.v + k1 * y
        self.p11 = self.p11 - k1 * self.p01
        self.p00, self.p01 = (1 - k0) * self.p00, (1 - k0) * self.p01

    def update_vel(self, z: np.ndarray, r: np.ndarray):
        s = self.p11 + r
        k0, k1 = self.p01 / s, self.p11 / s
        y = z - self.v
        self.x = self.x + k0 * y
        self.v = self.v + k1 * y
        self.p00 = self.p00 - k0 * self.p01
        self.p01, self.p11 = (1 - k1) * self.p01, (1 - k1) * self.p11


class _Track:
    __slots__ = ("kf", "gen", "last_fix_t", "hist_t", "hist", "out_seg", "out_t", "offset")

    def __init__(self):
        self.kf: Optional[_Kalman] = None
        self.gen = 0                    # bumped when the filter restarts
        self.last_fix_t = -math.inf
        self.hist_t: list = []          # filtered state times, ascending
        self.hist: list = []            # (gen, position, velocity) at those times
        self.out_seg: Optional[Segment] = None
        self.out_t = -math.inf
        self.offset = np.zeros(3)


class Tracker:
    """Per-aircraft live estimates. Used from the render thread only;
    ``configure`` may be called from anywhere and takes effect on the next frame."""

    def __init__(self, settings: Optional[dict] = None, coast_s: float = 30.0):
        self.settings = clean_settings(settings)
        self.coast_s = float(coast_s)
        self._tracks: dict = {}
        self._frame = None
        self._dirty = False
        self._latest_t = -math.inf

    def configure(self, settings: dict):
        self.settings = clean_settings(settings, self.settings)
        self._dirty = True

    def estimator(self, frame) -> Callable:
        """Call once per rendered frame; returns ``est(icao, fixes, t) -> ENU or None``."""
        if self._dirty or frame is not self._frame:
            self._tracks.clear()
            self._dirty = False
            self._frame = frame
        old = self._latest_t - FORGET_S
        for icao in [k for k, tr in self._tracks.items() if tr.out_t < old]:
            del self._tracks[icao]
        return self.estimate

    def estimate(self, icao: str, fixes: list, t: float) -> Optional[np.ndarray]:
        st, frame = self.settings, self._frame
        tr = self._tracks.get(icao)
        if tr is None:
            tr = self._tracks[icao] = _Track()
        if st["mode"] == "kalman":
            self._ingest(tr, fixes, frame, st)
            seg = self._kalman_segment(tr, t)
        else:
            seg = raw_segment(fixes, t, frame, self.coast_s)
        if seg is None:
            tr.out_seg = None
            return None
        self._latest_t = max(self._latest_t, t)
        return self._blend(tr, seg, t, st["blend_s"])

    # ── kalman ────────────────────────────────────────────────────────────
    def _ingest(self, tr: _Track, fixes: list, frame, st: dict):
        if not fixes or fixes[-1][T] <= tr.last_fix_t:
            return
        i = len(fixes)
        while i > 0 and fixes[i - 1][T] > tr.last_fix_t:
            i -= 1
        r_pos = np.array([st["pos_noise_m"] ** 2] * 2 + [min(st["pos_noise_m"], ALT_NOISE_M) ** 2])
        r_vel = np.full(3, st["vel_noise_ms"] ** 2)
        q = st["accel_noise"] ** 2
        for f in fixes[i:]:
            p = _fix_enu(frame, f)
            v = reported_velocity(frame, f, p) if st["use_velocity"] else None
            kf = tr.kf
            if kf is None or f[T] - kf.t > self.coast_s:
                tr.kf = _Kalman(f[T], p, v, r_pos, r_vel)
                tr.gen += 1
            else:
                kf.predict(f[T], q)
                kf.update_pos(p, r_pos)
                if v is not None:
                    kf.update_vel(v, r_vel)
            tr.last_fix_t = f[T]
            tr.hist_t.append(f[T])
            tr.hist.append((tr.gen, tr.kf.x.copy(), tr.kf.v.copy()))
        if len(tr.hist) > HISTORY:
            del tr.hist_t[:-HISTORY], tr.hist[:-HISTORY]

    def _kalman_segment(self, tr: _Track, t: float) -> Optional[Segment]:
        if not tr.hist:
            return None
        # The newest filtered state at or before t: the frame time usually trails
        # the newest fix by the sync offset, and a fix from its future isn't used.
        j = max(0, bisect_right(tr.hist_t, t) - 1)
        t0 = tr.hist_t[j]
        if abs(t - t0) > self.coast_s:
            return None
        gen, x, v = tr.hist[j]
        return Segment(("kf", gen, t0), t0, x, v)

    # ── easing ────────────────────────────────────────────────────────────
    @staticmethod
    def _blend(tr: _Track, seg: Segment, t: float, blend_s: float) -> np.ndarray:
        pos = seg.at(t)
        dt = t - tr.out_t
        if blend_s <= 0 or tr.out_seg is None or not (0 <= dt < BLEND_GAP_S):
            tr.offset = np.zeros(3)
        else:
            tr.offset = tr.offset * math.exp(-3.0 * dt / blend_s)
            if seg.key != tr.out_seg.key:
                # Keep the marker where it was; the difference fades out from here.
                jump = tr.out_seg.at(t) - pos
                tr.offset = tr.offset + jump if np.linalg.norm(jump) < SNAP_M else np.zeros(3)
        tr.out_seg, tr.out_t = seg, t
        return pos + tr.offset
