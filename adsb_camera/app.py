"""Core application: ties video, ADS-B, calibration and rendering together.

The HTTP layer (server.py) only calls methods on AdsbCamera.
"""
from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections import deque
from dataclasses import asdict
from typing import Optional

import cv2
import numpy as np

from .aircraft import AircraftStore, MqttFeed
from .autodetect import AutoRefiner
from .calibration import (CalibrationStore, Correspondence, Site, SolveOptions, evaluate,
                          predict, solve, trim_fixes)
from .camera_model import MODELS, CameraModel
from .logbook import LogStore
from .overlay import RETICLE_SHAPES, draw_aircraft, draw_grid, draw_hud, draw_points, placeholder
from .scene import compute_scene, fov_outline, sky_grid
from .video import VideoSource

log = logging.getLogger("adsb_camera")


def _num(v) -> Optional[float]:
    """Config numbers may arrive as '', None or strings (env vars)."""
    try:
        return None if v is None or str(v).strip() == "" else float(v)
    except ValueError:
        return None


class AdsbCamera:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        acfg, scfg = cfg["aircraft"], cfg["server"]
        os.makedirs(cfg["data_dir"], exist_ok=True)
        self.calib_store = CalibrationStore(os.path.join(cfg["data_dir"], "calibration.json"))
        saved = self.calib_store.load()

        s = cfg["site"]
        cfg_site = Site(_num(s.get("lat")), _num(s.get("lon")), _num(s.get("alt_m")) or 0.0,
                        _num(s.get("geoid_m")) or 0.0)
        self.site: Site = saved.get("site") if saved.get("site") and saved["site"].valid else cfg_site
        self.camera: CameraModel = saved.get("camera") or self._default_camera()
        self.points: list = saved.get("points", [])

        st = saved.get("settings", {})
        self.overlay = {**cfg["overlay"], **st.get("overlay", {})}
        self.solve_opts = {"model": "auto", "sync": "auto", "solve_position": False,
                           "lock_intrinsics": False, "auto_solve": True, **st.get("solve", {})}
        self.max_range_km = float(st.get("max_range_km", acfg["max_range_km"]))
        self.autodetect_enabled = bool(st.get("autodetect_enabled", cfg["autodetect"]["enabled"]))

        lb = cfg["logbook"]
        self.logstore = LogStore(os.path.join(cfg["data_dir"], "frames"), max_auto=int(lb["max_auto_frames"]))
        self.autolog_enabled = bool(st.get("autolog_enabled", lb["enabled"]))
        self._viewing_frame_id: Optional[str] = None
        self._autolog_buffer: deque = deque()      # (t, frame) sampled every interval_s
        self._autolog_last_buffer_t = 0.0
        self._autolog_last_log_t = 0.0
        self._autolog_active: dict = {}            # icao -> deadline (t) to keep logging until
        self._autolog_in_view: dict = {}           # icao -> last in_view, to catch the transition
        self._autolog_saved_ts: set = set()        # rounded frame times already saved, dedupe

        self.store = AircraftStore(acfg["altitude_source"], self.site.geoid_m,
                                   self.site.ellipsoid_alt, timeout_s=acfg["timeout_s"])
        m = cfg["mqtt"]
        self.feed = MqttFeed(self.store, m["host"], m["port"], m["prefix"], m["username"],
                             m["password"], bool(m["tls"]), client_id=f"adsb-camera-{os.getpid()}")
        v = cfg["video"]
        self.video = VideoSource(v["source"], v["rtsp_transport"], v["reconnect_s"],
                                 v.get("width"), v.get("height"), v.get("fps"),
                                 v.get("fourcc", "MJPG"), v.get("backend", "auto"))
        ad = cfg["autodetect"]
        self.refiner = AutoRefiner(ad["interval_s"], ad["search_px"])

        self.lock = threading.RLock()
        self._solve_lock = threading.Lock()   # one solve at a time
        self.version = 0
        self.frozen: Optional[dict] = None
        self.scene: list = []
        self.last_solve: dict = {"message": "Not calibrated yet.", "ok": None, "at": None}
        self.clients = 0
        self._jpeg: Optional[bytes] = None
        self._jpeg_seq = 0
        self._jpeg_t = 0.0
        self._rendered_version = -1
        self._grid_cache = (None, None)
        self._frame_cache = (None, None)
        self._pred: dict = {}
        self._auto_since_solve = 0
        self._last_auto_solve = 0.0
        self._stop = threading.Event()
        self._render_thread: Optional[threading.Thread] = None
        self._refresh_predictions()

    # ── lifecycle ─────────────────────────────────────────────────────────
    def start(self):
        self.video.start()
        self.feed.start()
        self._render_thread = threading.Thread(target=self._render_loop, name="render", daemon=True)
        self._render_thread.start()
        if not self.site.valid:
            log.warning("Camera location not set - set site.lat/lon in config.yaml or in the web UI")

    def stop(self):
        self._stop.set()
        self.feed.stop()
        self.video.stop()

    # ── helpers ───────────────────────────────────────────────────────────
    def _default_camera(self, w=1920, h=1080) -> CameraModel:
        c = self.cfg["camera"]
        model = c.get("model", "pinhole")
        if model not in MODELS:
            model = "pinhole"
        return CameraModel.default(w, h, model, float(c.get("hfov_deg") or 70),
                                   float(c.get("yaw_deg") or 0), float(c.get("pitch_deg") or 10),
                                   float(c.get("roll_deg") or 0))

    def _changed(self, save=True):
        self.version += 1
        self._refresh_predictions()
        if save:
            self._save()

    def _save(self):
        try:
            self.calib_store.save(self.site, self.camera, self.points, {
                "overlay": self.overlay, "solve": self.solve_opts,
                "max_range_km": self.max_range_km, "autodetect_enabled": self.autodetect_enabled,
                "autolog_enabled": self.autolog_enabled})
        except OSError as e:
            log.error("Could not save calibration: %s", e)

    def _refresh_predictions(self):
        with self.lock:
            self._pred = predict(self.camera, self.points, self.site)

    def _site_frame(self):
        key = (self.site.lat, self.site.lon, self.site.ellipsoid_alt)
        if self._frame_cache[0] != key:
            self._frame_cache = (key, self.site.frame() if self.site.valid else None)
        return self._frame_cache[1]

    def _grid(self, cam: CameraModel):
        key = (id(cam), self.version)
        if self._grid_cache[0] != key:
            self._grid_cache = (key, sky_grid(cam))
        return self._grid_cache[1]

    def latest_jpeg(self):
        return self._jpeg, self._jpeg_seq

    # ── render loop ───────────────────────────────────────────────────────
    def _render_loop(self):
        period = 1.0 / max(1, float(self.cfg["server"]["output_fps"]))
        while not self._stop.is_set():
            t0 = time.time()
            try:
                self._render_once()
            except Exception:  # never let the loop die
                log.exception("render error")
                time.sleep(1.0)
            self._stop.wait(max(0.005, period - (time.time() - t0)))

    def _render_once(self):
        acfg, scfg = self.cfg["aircraft"], self.cfg["server"]
        # Read the frame and the version stamp together. Loading a saved frame sets
        # both; picking them up from either side of that would stamp the new version
        # onto the old picture, and the viewer would sit on the previous frame.
        with self.lock:
            frozen, version = self.frozen, self.version
        if frozen:
            frame, t, snap = frozen["frame"], frozen["t"], frozen["snapshot"]
        else:
            frame, t, _ = self.video.latest()
            snap = self.store.snapshot(since=time.time() - acfg["trail_s"] - 60)

        want_image = self.clients > 0 or time.time() - self._jpeg_t > 2.0
        if frame is None:
            if want_image:
                msg = f"Waiting for video: {self.video.describe()}"
                if self.video.error:
                    msg += f"  ({self.video.error})"
                self._publish(placeholder(1280, 720, msg))
            return

        h, w = frame.shape[:2]
        with self.lock:
            if (w, h) != (self.camera.width, self.camera.height):
                log.info("Frame size %dx%d - adapting camera model", w, h)
                self.camera = self.camera.scaled_to(w, h)
                self._changed()
            cam = self.camera
            opts = dict(self.overlay)

        frame_enu = self._site_frame()
        scene = [] if frame_enu is None else compute_scene(
            cam, frame_enu, snap, t, self.max_range_km, acfg["stale_s"], acfg["trail_s"],
            with_trails=want_image and opts.get("trails", True))
        self.scene = scene

        if (self.autodetect_enabled and not frozen and cam.calibrated
                and math.isfinite(cam.rms_px) and cam.rms_px < 10):
            self._autodetect(frame, t, scene)

        if self.autolog_enabled and not frozen and frame_enu is not None:
            self._autolog(frame, t, scene)

        if not want_image or (frozen and version == self._rendered_version):
            return
        self._rendered_version = version if frozen else -1

        out_w = min(w, int(scfg["output_max_width"]))
        s = out_w / w
        img = cv2.resize(frame, (out_w, int(round(h * s))), interpolation=cv2.INTER_AREA) \
            if s < 1 else frame.copy()
        ui = max(0.6, out_w / 1280)
        if opts.get("grid"):
            draw_grid(img, self._grid(cam), s, ui)
        if opts.get("points") or frozen:
            draw_points(img, self._points_for_overlay(cam), s, ui)
        draw_aircraft(img, scene, s, ui, opts)
        if opts.get("hud"):
            draw_hud(img, self._hud_lines(cam, scene), ui, frozen=bool(frozen))
        elif frozen:
            draw_hud(img, [], ui, frozen=True)
        self._publish(img)

    def _publish(self, img):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(self.cfg["server"]["jpeg_quality"])])
        if ok:
            self._jpeg = buf.tobytes()
            self._jpeg_seq += 1
            self._jpeg_t = time.time()

    def _points_for_overlay(self, cam):
        with self.lock:
            pts, pred = list(self.points), dict(self._pred)
        out = []
        for p in pts:
            out.append({"obs": p.pixel_for(cam.width, cam.height), "pred": pred.get(p.id),
                        "enabled": p.enabled})
        return out

    def _hud_lines(self, cam, scene):
        n_view = sum(1 for a in scene if a["in_view"])
        mq = "ADS-B ok" if self.feed.connected else "ADS-B offline"
        cal = (f"{cam.model} {cam.hfov():.0f} deg FOV, RMS {cam.rms_px:.1f}px, {len(self.points)} pts"
               if cam.calibrated else f"uncalibrated ({len(self.points)} pts)")
        return [f"{mq} | {len(scene)} aircraft, {n_view} in view | {cal}"]

    # ── auto refinement ───────────────────────────────────────────────────
    def _autodetect(self, frame, t, scene):
        found = self.refiner.process(frame, t, scene,
                                     max_offset_px=max(4.0 * self.camera.rms_px, 8.0))
        if not found:
            return
        h, w = frame.shape[:2]
        cap = int(self.cfg["autodetect"]["max_points"])
        with self.lock:
            for icao, (u, v) in found:
                fixes = trim_fixes(self.store.fixes_for(icao), t)
                if not fixes:
                    continue
                a = next((x for x in scene if x["icao"] == icao), None)
                self.points.append(Correspondence(u=u, v=v, img_w=w, img_h=h, t=t, icao=icao,
                                                  label=(a or {}).get("callsign") or icao,
                                                  fixes=fixes, source="auto"))
                self._auto_since_solve += 1
            autos = [p for p in self.points if p.source == "auto"]
            for p in autos[: max(0, len(autos) - cap)]:
                self.points.remove(p)
            self._changed()
        every = float(self.cfg["autodetect"]["resolve_every_s"])
        if self._auto_since_solve >= 3 and time.time() - self._last_auto_solve > every:
            self._last_auto_solve = time.time()
            self._auto_since_solve = 0
            threading.Thread(target=self.run_solve, daemon=True).start()

    # ── auto-log: save frames while a track transits the FOV ─────────────
    def _autolog(self, frame, t, scene):
        cfg = self.cfg["logbook"]
        interval, window = float(cfg["interval_s"]), float(cfg["window_s"])

        # Keep a rolling buffer so an entry can be logged a few seconds "before" it happened.
        buf = self._autolog_buffer
        if t - self._autolog_last_buffer_t >= interval:
            self._autolog_last_buffer_t = t
            buf.append((t, frame.copy()))
            cutoff = t - window - interval
            while buf and buf[0][0] < cutoff:
                buf.popleft()

        entered = []
        seen = set()
        for a in scene:
            icao = a["icao"]
            seen.add(icao)
            was_in_view = self._autolog_in_view.get(icao, False)
            if a["in_view"]:
                if not was_in_view:
                    entered.append(icao)
                # Refresh every tick it's in view, so logging spans the whole transit
                # (plus `window` seconds either side), not just the moment it entered.
                self._autolog_active[icao] = t + window
            self._autolog_in_view[icao] = a["in_view"]
        for icao in list(self._autolog_in_view):
            if icao not in seen:
                self._autolog_in_view.pop(icao, None)

        if entered:
            for bt, bframe in list(buf):
                key = round(bt, 1)
                if key in self._autolog_saved_ts:
                    continue
                self._autolog_saved_ts.add(key)
                self.logstore.add(bframe, bt, self.store.snapshot(since=bt - 150),
                                  source="auto", reason="entering view")

        self._autolog_active = {k: v for k, v in self._autolog_active.items() if v >= t}
        if self._autolog_active and t - self._autolog_last_log_t >= interval:
            self._autolog_last_log_t = t   # advance regardless of dedup, to keep a steady cadence
            key = round(t, 1)
            if key not in self._autolog_saved_ts:
                self._autolog_saved_ts.add(key)
                self.logstore.add(frame.copy(), t, self.store.snapshot(since=t - 150),
                                  source="auto", reason="in view")
        if len(self._autolog_saved_ts) > 4096:
            self._autolog_saved_ts = set(sorted(self._autolog_saved_ts)[-2048:])

    # ── freeze / points ───────────────────────────────────────────────────
    def freeze(self) -> dict:
        frame, t, _ = self.video.latest()
        if frame is None:
            return {"ok": False, "message": "No video frame yet."}
        with self.lock:
            self.frozen = {"frame": frame.copy(), "t": t, "snapshot": self.store.snapshot()}
            self._viewing_frame_id = None
            self.version += 1
        return {"ok": True, "t": t}

    def unfreeze(self):
        with self.lock:
            self.frozen = None
            self._viewing_frame_id = None
            self.version += 1
        return {"ok": True}

    # ── saved frames (manual + auto-logged) ───────────────────────────────
    def save_frame(self) -> dict:
        frame, t, _ = self.video.latest()
        if frame is None:
            return {"ok": False, "message": "No video frame yet."}
        entry = self.logstore.add(frame, t, self.store.snapshot(since=t - 150),
                                  source="manual", reason="manual")
        return {"ok": True, "id": entry["id"], "t": t}

    def list_frames(self) -> dict:
        with self.lock:
            viewing = self._viewing_frame_id
        return {"ok": True, "items": self.logstore.list(), "viewing": viewing}

    def view_frame(self, fid: str) -> dict:
        full = self.logstore.get_full(fid)
        if not full:
            return {"ok": False, "message": "No such saved frame."}
        img = self.logstore.load_image(fid)
        if img is None:
            return {"ok": False, "message": "Saved frame image is missing."}
        with self.lock:
            self.frozen = {"frame": img, "t": full["t"], "snapshot": full["snapshot"]}
            self._viewing_frame_id = fid
            self.version += 1
        return {"ok": True, "t": full["t"]}

    def delete_frame(self, fid: str) -> dict:
        removed = self.logstore.delete(fid)
        with self.lock:
            if self._viewing_frame_id == fid:
                self.frozen = None
                self._viewing_frame_id = None
                self.version += 1
        return {"ok": removed}

    def clear_frames(self) -> dict:
        n = self.logstore.clear()
        with self.lock:
            if self._viewing_frame_id is not None:
                self.frozen = None
                self._viewing_frame_id = None
                self.version += 1
        return {"ok": True, "deleted": n}

    def frozen_jpeg(self) -> Optional[bytes]:
        with self.lock:
            fr = self.frozen
        if not fr:
            return None
        ok, buf = cv2.imencode(".jpg", fr["frame"], [cv2.IMWRITE_JPEG_QUALITY, 92])
        return buf.tobytes() if ok else None

    def candidates(self, x: float, y: float, limit: int = 12) -> dict:
        with self.lock:
            fr, cam = self.frozen, self.camera
        if not fr:
            return {"ok": False, "message": "Freeze the frame first."}
        frame_enu = self._site_frame()
        if frame_enu is None:
            return {"ok": False, "message": "Set the camera location first."}
        u, v = x * cam.width, y * cam.height
        scene = compute_scene(cam, frame_enu, fr["snapshot"], fr["t"], self.max_range_km,
                              stale_s=60, trail_s=0, with_trails=False)
        used = {p.icao for p in self.points if p.kind == "aircraft" and abs(p.t - fr["t"]) < 0.01}
        items = []
        for a in scene:
            if a["el"] < -3:
                continue
            d = math.hypot(a["uv"][0] - u, a["uv"][1] - v) if a["uv"] else None
            items.append({"icao": a["icao"], "callsign": a["callsign"], "alt_ft": a["alt_ft"],
                          "range_km": round(a["range_km"], 1), "az": round(a["az"], 1),
                          "el": round(a["el"], 1), "dist_px": None if d is None else round(d),
                          "age_s": round(a["age_s"], 1), "used": a["icao"] in used})
        if cam.calibrated or len(self.points) >= 2:
            items.sort(key=lambda i: (i["dist_px"] is None, i["dist_px"] or 0))
        else:
            items.sort(key=lambda i: i["range_km"])
        return {"ok": True, "calibrated": cam.calibrated, "items": items[:limit]}

    def add_point(self, x: float, y: float, icao: Optional[str] = None,
                  landmark: Optional[dict] = None) -> dict:
        with self.lock:
            fr, cam = self.frozen, self.camera
            if icao:
                if not fr:
                    return {"ok": False, "message": "Freeze the frame first."}
                ac = fr["snapshot"].get(icao.upper())
                if not ac:
                    return {"ok": False, "message": f"{icao} not in the frozen snapshot."}
                fixes = trim_fixes(ac["fixes"], fr["t"])
                p = Correspondence(u=x * cam.width, v=y * cam.height, img_w=cam.width,
                                   img_h=cam.height, t=fr["t"], icao=icao.upper(),
                                   label=ac.get("callsign") or icao.upper(), fixes=fixes)
                p.merge_fixes(self.store.fixes_for(icao))
            elif landmark:
                try:
                    lat, lon = float(landmark["lat"]), float(landmark["lon"])
                    alt = float(landmark.get("alt_m") or 0.0)
                except (KeyError, TypeError, ValueError):
                    return {"ok": False, "message": "Landmark needs numeric lat, lon and alt_m."}
                p = Correspondence(u=x * cam.width, v=y * cam.height, img_w=cam.width,
                                   img_h=cam.height, t=fr["t"] if fr else time.time(),
                                   kind="landmark", label=landmark.get("label") or "landmark",
                                   lat=lat, lon=lon, alt_m=alt)
            else:
                return {"ok": False, "message": "Pick an aircraft or enter a landmark."}
            self.points.append(p)
            self._changed()
            n = sum(1 for q in self.points if q.enabled)
        if self.solve_opts.get("auto_solve") and n >= 2:
            return {"ok": True, "id": p.id, "solve": self.run_solve()}
        self._update_residuals()
        return {"ok": True, "id": p.id}

    def update_point(self, pid: str, enabled: Optional[bool] = None) -> dict:
        with self.lock:
            p = next((q for q in self.points if q.id == pid), None)
            if not p:
                return {"ok": False, "message": "No such point."}
            if enabled is not None:
                p.enabled = bool(enabled)
            self._changed()
        self._update_residuals()
        return {"ok": True}

    def delete_points(self, pid: Optional[str] = None, source: Optional[str] = None) -> dict:
        with self.lock:
            before = len(self.points)
            if pid:
                self.points = [p for p in self.points if p.id != pid]
            elif source:
                self.points = [p for p in self.points if p.source != source]
            else:
                self.points = []
            self._changed()
            removed = before - len(self.points)
        self._update_residuals()
        return {"ok": True, "removed": removed}

    # ── solving / manual camera ───────────────────────────────────────────
    def run_solve(self, overrides: Optional[dict] = None) -> dict:
        with self._solve_lock:
            return self._run_solve(overrides)

    def _run_solve(self, overrides: Optional[dict] = None) -> dict:
        with self.lock:
            opts = {**self.solve_opts, **(overrides or {})}
            cam, site = self.camera.copy(), Site(**asdict(self.site))
            for p in self.points:
                if p.kind == "aircraft" and time.time() - p.t < 600:
                    p.merge_fixes(self.store.fixes_for(p.icao))
            pts = list(self.points)
        opt = SolveOptions(model=opts.get("model", "auto"), sync=opts.get("sync", "auto"),
                           solve_position=bool(opts.get("solve_position")),
                           lock_intrinsics=bool(opts.get("lock_intrinsics")))
        try:
            res = solve(cam, pts, site, opt)
            # Auto-added points that the fit rejects are dropped (once);
            # manual points stay for the user to judge.
            bad_auto = {p.id for p in pts if p.source == "auto"} & set(res.get("outliers") or [])
            if res.get("ok") and bad_auto:
                log.info("Dropping %d auto point(s) that don't fit", len(bad_auto))
                with self.lock:
                    self.points = [p for p in self.points if p.id not in bad_auto]
                    pts = list(self.points)
                res = solve(cam, pts, site, opt)
                if res.get("ok"):
                    res["message"] += f"; dropped {len(bad_auto)} bad auto point(s)"
        except Exception as e:  # keep the service up whatever the data does
            log.exception("solve failed")
            res = {"ok": False, "message": f"Solver error: {e}"}
        with self.lock:
            if res.get("ok"):
                new = res["camera"]
                if (new.width, new.height) == (self.camera.width, self.camera.height):
                    self.camera = new
                if res.get("site"):
                    self.site = res["site"]
                for p in self.points:
                    if p.id in res["residuals"]:
                        p.residual_px = res["residuals"][p.id]
                self._changed()
            self.last_solve = {"ok": res.get("ok"), "message": res.get("message"), "at": time.time(),
                               "free": res.get("free"), "outliers": res.get("outliers", []),
                               "alternatives": res.get("alternatives")}
        return self.last_solve

    def _update_residuals(self):
        with self.lock:
            res = evaluate(self.camera, self.points, self.site)
            for p in self.points:
                p.residual_px = res.get(p.id)
            e = np.array([r for r in res.values() if r is not None], float)
            if len(e):
                self.camera.rms_px = float(np.sqrt(np.mean(e ** 2)))

    def set_camera(self, d: dict) -> dict:
        with self.lock:
            cam = self.camera.copy()
            if d.get("model") in MODELS and d["model"] != cam.model:
                cam = cam.with_model(d["model"])
            if any(k in d for k in ("yaw", "pitch", "roll")):
                y, p, r = cam.ypr()
                cam.set_ypr(float(d.get("yaw", y)), float(d.get("pitch", p)), float(d.get("roll", r)))
            if "hfov" in d:
                target = max(5.0, min(float(d["hfov"]), 170.0 if cam.model == "pinhole" else 260.0))
                for _ in range(8):  # distortion-aware: iterate f until the FOV matches
                    cur = cam.hfov()
                    if not math.isfinite(cur) or abs(cur - target) < 0.01:
                        break
                    if cam.model == "pinhole":
                        cam.f *= math.tan(math.radians(cur / 2)) / math.tan(math.radians(target / 2))
                    else:
                        cam.f *= cur / target
            if "sync_offset_s" in d:
                cam.sync_offset_s = float(d["sync_offset_s"])
            if "cx" in d:
                cam.cx = float(d["cx"]) * cam.width
            if "cy" in d:
                cam.cy = float(d["cy"]) * cam.height
            self.camera = cam
            self._changed()
        self._update_residuals()
        return {"ok": True}

    def reset_camera(self) -> dict:
        with self.lock:
            self.camera = self._default_camera(self.camera.width, self.camera.height)
            self.last_solve = {"message": "Camera reset to the starting guess.", "ok": None, "at": time.time()}
            self._changed()
        self._update_residuals()
        return {"ok": True}

    def set_site(self, d: dict) -> dict:
        try:
            site = Site(float(d["lat"]), float(d["lon"]), float(d.get("alt_m") or 0.0),
                        float(d.get("geoid_m", self.site.geoid_m) or 0.0))
        except (KeyError, TypeError, ValueError):
            return {"ok": False, "message": "lat and lon must be numbers."}
        if not (-90 <= site.lat <= 90 and -180 <= site.lon <= 180):
            return {"ok": False, "message": "lat/lon out of range."}
        with self.lock:
            self.site = site
            self.store.geoid_m = site.geoid_m
            self.store.ground_alt_m = site.ellipsoid_alt
            self._changed()
        self._update_residuals()
        return {"ok": True}

    def set_settings(self, d: dict) -> dict:
        with self.lock:
            for k, v in (d.get("overlay") or {}).items():
                if k == "reticle_shape":
                    if v in RETICLE_SHAPES:
                        self.overlay[k] = v
                elif k == "reticle_size":
                    try:
                        self.overlay[k] = min(4.0, max(0.3, float(v)))
                    except (TypeError, ValueError):
                        pass
                elif k in self.overlay:
                    self.overlay[k] = bool(v)
            for k, v in (d.get("solve") or {}).items():
                if k in self.solve_opts:
                    self.solve_opts[k] = v
            if "max_range_km" in d:
                self.max_range_km = max(1.0, float(d["max_range_km"]))
            if "autodetect" in d:
                self.autodetect_enabled = bool(d["autodetect"])
            if "autolog" in d:
                self.autolog_enabled = bool(d["autolog"])
            self._changed()
        return {"ok": True}

    # ── state for the UI ──────────────────────────────────────────────────
    def state(self) -> dict:
        with self.lock:
            cam, site, frozen = self.camera, self.site, self.frozen
            pts = [p.summary(cam.width, cam.height) for p in self.points]
            outliers = set(self.last_solve.get("outliers") or [])
            for p in pts:
                p["outlier"] = p["id"] in outliers
            settings = {"overlay": dict(self.overlay), "solve": dict(self.solve_opts),
                        "max_range_km": self.max_range_km, "autodetect": self.autodetect_enabled,
                        "autolog": self.autolog_enabled}
            last_solve = dict(self.last_solve)
            viewing_frame = self._viewing_frame_id
        aircraft = []
        for a in self.scene:
            aircraft.append({k: a[k] for k in ("icao", "callsign", "alt_ft", "on_ground", "gs_kt",
                                                "az", "el", "range_km", "in_view", "emergency")}
                            | {"x": a["uv"][0] / cam.width if a["uv"] else None,
                               "y": a["uv"][1] / cam.height if a["uv"] else None,
                               "rgb": a["rgb"]})
        frame, _, _ = self.video.latest()
        return {
            "camera": cam.to_dict(), "site": asdict(site), "site_valid": site.valid,
            "fov_outline": fov_outline(cam), "points": pts, "last_solve": last_solve,
            "aircraft": aircraft, "frozen": bool(frozen),
            "frozen_t": frozen["t"] if frozen else None, "viewing_frame": viewing_frame,
            "settings": settings,
            "video": {"connected": self.video.connected, "fps": round(self.video.fps, 1),
                      "source": self.video.describe(), "error": self.video.error,
                      "format": self.video.format,
                      "width": frame.shape[1] if frame is not None else None,
                      "height": frame.shape[0] if frame is not None else None},
            "mqtt": {"connected": self.feed.connected, "host": f"{self.feed.host}:{self.feed.port}",
                     "error": self.feed.last_error, "tracked": len(self.store)},
            "autodetect_stats": dict(self.refiner.stats),
            "server_time": time.time(),
        }
