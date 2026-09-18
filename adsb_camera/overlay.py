"""Draw the ADS-B overlay onto a (possibly downscaled) frame with OpenCV."""
from __future__ import annotations

import math

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
GRID_BGR = (235, 225, 210)
POINT_OBS = (90, 230, 90)
POINT_PRED = (80, 80, 255)
POINT_OFF = (150, 150, 150)


def _bgr(rgb):
    return (int(rgb[2]), int(rgb[1]), int(rgb[0]))


def _text(img, txt, org, scale, color, thick=1):
    """Outlined text. A trailing 'deg' is drawn as a real degree ring
    (Hershey fonts have no degree glyph)."""
    x, y = int(org[0]), int(org[1])
    deg = txt.endswith("deg")
    if deg:
        txt = txt[:-3]
    cv2.putText(img, txt, (x, y), FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, txt, (x, y), FONT, scale, color, thick, cv2.LINE_AA)
    if deg:
        (tw, th), _ = cv2.getTextSize(txt, FONT, scale, thick)
        c, r = (x + tw + int(4 * scale) + 1, y - th + int(3 * scale)), max(2, int(4 * scale))
        cv2.circle(img, c, r, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.circle(img, c, r, color, thick, cv2.LINE_AA)


def _alt_text(a) -> str:
    if a["on_ground"]:
        return "GND"
    ft = a["alt_ft"]
    if ft is None:
        return "---"
    return f"FL{ft / 100:03.0f}" if ft >= 7000 else f"{ft:,.0f}ft"


def _vrate_mark(a) -> str:
    vr = a.get("vrate_fpm")
    if not isinstance(vr, (int, float)) or abs(vr) < 300:
        return ""
    return " ^" if vr > 0 else " v"


def draw_grid(img, grid, s, ui):
    th = max(1, int(round(ui)))
    layer = img.copy()
    for el, lines in grid["rings"].items():
        for ln in lines:
            pts = (ln * s).astype(np.int32)
            cv2.polylines(layer, [pts], False, GRID_BGR, th * (2 if el == 0 else 1), cv2.LINE_AA)
    for az, lines in grid["spokes"].items():
        if az % 30:
            continue
        for ln in lines:
            cv2.polylines(layer, [(ln * s).astype(np.int32)], False, GRID_BGR, th, cv2.LINE_AA)
    cv2.addWeighted(layer, 0.35, img, 0.65, 0, dst=img)
    for uv, txt, major in grid["labels"]:
        _text(img, txt, (uv[0] * s + 4, uv[1] * s - 6), 0.45 * ui * (1.25 if major else 1.0),
              GRID_BGR, th)


def _rects_overlap(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def draw_aircraft(img, scene, s, ui, opts):
    h, w = img.shape[:2]
    placed = []
    th = max(1, int(round(ui)))
    fs = 0.5 * ui
    lh = int(18 * ui)
    # Nearest last so they draw on top.
    for a in sorted(scene, key=lambda a: -a["range_km"]):
        color = _bgr(a["rgb"])
        if a["emergency"]:
            color = (0, 0, 255)
        if opts.get("trails") and a["trail"]:
            prev = None
            n = len(a["trail"])
            for i, q in enumerate(a["trail"]):
                if q is None:
                    prev = None
                    continue
                q = (int(q[0] * s), int(q[1] * s))
                if prev is not None:
                    fade = 0.25 + 0.75 * i / max(1, n - 1)
                    c = tuple(int(ch * fade) for ch in color)
                    cv2.line(img, prev, q, c, th, cv2.LINE_AA)
                prev = q

    for a in sorted(scene, key=lambda a: a["range_km"]):
        if a["uv"] is None:
            continue
        color = (0, 0, 255) if a["emergency"] else _bgr(a["rgb"])
        u, v = a["uv"][0] * s, a["uv"][1] * s
        if a["in_view"]:
            r = int(11 * ui)
            c = (int(u), int(v))
            cv2.circle(img, c, r, (0, 0, 0), th + 2, cv2.LINE_AA)
            cv2.circle(img, c, r, color, th, cv2.LINE_AA)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                p1 = (int(u + dx * r * 1.4), int(v + dy * r * 1.4))
                p2 = (int(u + dx * r * 2.1), int(v + dy * r * 2.1))
                cv2.line(img, p1, p2, color, th, cv2.LINE_AA)
            if not opts.get("labels"):
                continue
            l1 = (a["callsign"] or a["icao"]) + _vrate_mark(a)
            gs = a["gs_kt"]
            l2 = f"{_alt_text(a)}  {gs:.0f}kt  {a['range_km']:.0f}km" if isinstance(gs, (int, float)) \
                else f"{_alt_text(a)}  {a['range_km']:.0f}km"
            tw = max(cv2.getTextSize(t, FONT, fs, th)[0][0] for t in (l1, l2))
            # Try right, lower-right, upper-right, left... first free slot.
            # Nearest aircraft are labelled first; if nothing is free, skip.
            for ox, oy in ((1, -1), (1, 1), (1, -3), (-1, -1), (-1, 1), (1, 3), (-1, -3)):
                x0 = u + (r * 2.4 if ox > 0 else -r * 2.4 - tw)
                y0 = v + oy * lh * 0.5
                rect = (x0 - 2, y0 - lh, x0 + tw + 2, y0 + lh * 1.3)
                if all(not _rects_overlap(rect, p) for p in placed):
                    break
            else:
                continue
            placed.append(rect)
            x0 = min(max(2, x0), w - tw - 2)
            y0 = min(max(lh, y0), h - lh)
            _text(img, l1, (x0, y0), fs * 1.1, color, th)
            _text(img, l2, (x0, y0 + lh), fs * 0.9, (235, 235, 235), th)
        elif opts.get("offscreen") and math.isfinite(u) and math.isfinite(v):
            # Edge arrow pointing towards an aircraft just outside the frame.
            inset = 16 * ui
            cu, cv_ = w / 2, h / 2
            dx, dy = u - cu, v - cv_
            k = min((w / 2 - inset) / abs(dx) if dx else 1e9, (h / 2 - inset) / abs(dy) if dy else 1e9)
            if k >= 1:
                continue
            ex, ey = cu + dx * k, cv_ + dy * k
            ang = math.atan2(dy, dx)
            size = 10 * ui
            tri = np.array([
                [ex + math.cos(ang) * size, ey + math.sin(ang) * size],
                [ex + math.cos(ang + 2.5) * size, ey + math.sin(ang + 2.5) * size],
                [ex + math.cos(ang - 2.5) * size, ey + math.sin(ang - 2.5) * size]], np.int32)
            cv2.fillPoly(img, [tri], color, cv2.LINE_AA)
            if opts.get("labels"):
                txt = a["callsign"] or a["icao"]
                tw = cv2.getTextSize(txt, FONT, fs * 0.8, th)[0][0]
                tx = min(max(4, ex - math.cos(ang) * size * 2.2 - tw / 2), w - tw - 4)
                ty = min(max(lh, ey - math.sin(ang) * size * 2.2 + 5), h - 6)
                _text(img, txt, (tx, ty), fs * 0.8, color, th)


def draw_points(img, points, s, ui):
    th = max(1, int(round(ui)))
    r = int(7 * ui)
    for p in points:
        o = (int(p["obs"][0] * s), int(p["obs"][1] * s))
        col = POINT_OBS if p["enabled"] else POINT_OFF
        if p.get("pred") is not None and p["enabled"]:
            q = (int(p["pred"][0] * s), int(p["pred"][1] * s))
            cv2.line(img, o, q, POINT_PRED, th, cv2.LINE_AA)
            cv2.circle(img, q, max(2, r // 2), POINT_PRED, -1, cv2.LINE_AA)
        cv2.drawMarker(img, o, (0, 0, 0), cv2.MARKER_TILTED_CROSS, 2 * r, th + 2, cv2.LINE_AA)
        cv2.drawMarker(img, o, col, cv2.MARKER_TILTED_CROSS, 2 * r, th, cv2.LINE_AA)


def draw_hud(img, lines, ui, frozen=False):
    th = max(1, int(round(ui)))
    y = int(24 * ui)
    for ln in lines:
        _text(img, ln, (int(10 * ui), y), 0.5 * ui, (240, 240, 240), th)
        y += int(20 * ui)
    if frozen:
        h, w = img.shape[:2]
        cv2.rectangle(img, (1, 1), (w - 2, h - 2), (70, 180, 255), max(2, th * 3))
        _text(img, "FROZEN - click aircraft to add calibration points",
              (int(10 * ui), h - int(12 * ui)), 0.6 * ui, (70, 180, 255), th)


def placeholder(width, height, message):
    img = np.full((height, width, 3), 40, np.uint8)
    ui = width / 1280
    _text(img, message, (int(30 * ui), height // 2), 0.8 * ui, (220, 220, 220), max(1, int(ui)))
    return img
