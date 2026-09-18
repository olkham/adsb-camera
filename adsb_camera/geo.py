"""Geodesy helpers (WGS-84 geodetic -> ECEF -> local ENU, az/el/range).

All angles in degrees at the API boundary, metres for distances.
ENU = East, North, Up, centred on the camera.
"""
from __future__ import annotations

import math

import numpy as np

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
EARTH_R = 6371000.0

FT_TO_M = 0.3048
KT_TO_MS = 0.514444
FPM_TO_MS = FT_TO_M / 60.0


def geodetic_to_ecef(lat, lon, h):
    lat = np.radians(np.asarray(lat, dtype=float))
    lon = np.radians(np.asarray(lon, dtype=float))
    h = np.asarray(h, dtype=float)
    s, c = np.sin(lat), np.cos(lat)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * s * s)
    x = (n + h) * c * np.cos(lon)
    y = (n + h) * c * np.sin(lon)
    z = (n * (1.0 - WGS84_E2) + h) * s
    return np.stack([x, y, z], axis=-1)


def ecef_to_enu_matrix(lat0: float, lon0: float) -> np.ndarray:
    la, lo = math.radians(lat0), math.radians(lon0)
    sl, cl = math.sin(la), math.cos(la)
    so, co = math.sin(lo), math.cos(lo)
    return np.array([
        [-so, co, 0.0],
        [-sl * co, -sl * so, cl],
        [cl * co, cl * so, sl],
    ])


class LocalFrame:
    """ENU frame anchored at the camera. Cheap to reuse for many conversions."""

    def __init__(self, lat: float, lon: float, alt_m: float):
        self.lat, self.lon, self.alt = float(lat), float(lon), float(alt_m)
        self.origin = geodetic_to_ecef(self.lat, self.lon, self.alt)
        self.M = ecef_to_enu_matrix(self.lat, self.lon)

    def enu(self, lat, lon, alt_m) -> np.ndarray:
        """Geodetic (scalars or arrays) -> ENU (...,3) metres."""
        ecef = geodetic_to_ecef(lat, lon, alt_m)
        return (ecef - self.origin) @ self.M.T


def az_el_range(enu: np.ndarray):
    """ENU (...,3) -> azimuth deg (0=N, clockwise), elevation deg, slant range m."""
    enu = np.asarray(enu, dtype=float)
    e, n, u = enu[..., 0], enu[..., 1], enu[..., 2]
    horiz = np.hypot(e, n)
    az = (np.degrees(np.arctan2(e, n)) + 360.0) % 360.0
    el = np.degrees(np.arctan2(u, horiz))
    rng = np.sqrt(horiz * horiz + u * u)
    return az, el, rng


def az_el_to_enu(az_deg, el_deg) -> np.ndarray:
    az = np.radians(np.asarray(az_deg, dtype=float))
    el = np.radians(np.asarray(el_deg, dtype=float))
    ce = np.cos(el)
    return np.stack([ce * np.sin(az), ce * np.cos(az), np.sin(el)], axis=-1)


def destination(lat: float, lon: float, bearing_deg: float, dist_m: float):
    """Great-circle forward projection (spherical, fine for dead reckoning)."""
    if dist_m == 0:
        return lat, lon
    d = dist_m / EARTH_R
    b = math.radians(bearing_deg)
    p1, l1 = math.radians(lat), math.radians(lon)
    sp2 = math.sin(p1) * math.cos(d) + math.cos(p1) * math.sin(d) * math.cos(b)
    p2 = math.asin(max(-1.0, min(1.0, sp2)))
    l2 = l1 + math.atan2(math.sin(b) * math.sin(d) * math.cos(p1),
                         math.cos(d) - math.sin(p1) * sp2)
    return math.degrees(p2), ((math.degrees(l2) + 540.0) % 360.0) - 180.0
