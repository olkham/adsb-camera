"""Live aircraft state from the adsb2mqtt topic tree.

Every lat/lon fix is timestamped with *our* wall clock on receipt, so video
frames and ADS-B positions share one time base. The constant offset between
"frame received" and "fix received" is a calibration parameter (sync offset).

A fix is stored as a plain list so it serialises straight into calibration
JSON:  [t, lat, lon, alt_m, gs_ms, track_deg, vrate_ms]  (NaN = unknown)
"""
from __future__ import annotations

import logging
import math
import threading
import time
from bisect import bisect_right
from collections import deque
from typing import Callable, Optional

from .geo import FPM_TO_MS, FT_TO_M, KT_TO_MS, destination

log = logging.getLogger("adsb_camera.aircraft")

NAN = float("nan")
T, LAT, LON, ALT, GS, TRK, VR = range(7)

NUMERIC = {
    "lat", "lon", "altitude_geom_ft", "altitude_m", "ground_speed_kt",
    "ground_speed_ms", "true_heading", "track", "vertical_rate_fpm",
    "nic", "nac_p", "messages", "rssi", "seen", "range_km", "bearing_deg",
}


def _finite(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def interp_track(fixes, t: float, max_extrapolate_s: float = 20.0):
    """Position (lat, lon, alt_m) of a track at time t, or None.

    Interpolates between bracketing fixes; dead-reckons past the last fix using
    ground speed / track / vertical rate (falling back to the velocity between
    the last two fixes). Returns None when t is too far outside the track.
    """
    if not fixes:
        return None
    times = [f[T] for f in fixes]
    i = bisect_right(times, t)
    if 0 < i < len(fixes):
        a, b = fixes[i - 1], fixes[i]
        span = b[T] - a[T]
        w = 0.0 if span <= 0 else (t - a[T]) / span
        dlon = ((b[LON] - a[LON] + 540.0) % 360.0) - 180.0
        return (a[LAT] + w * (b[LAT] - a[LAT]),
                a[LON] + w * dlon,
                a[ALT] + w * (b[ALT] - a[ALT]))

    ref = fixes[-1] if i == len(fixes) else fixes[0]
    dt = t - ref[T]
    if abs(dt) > max_extrapolate_s:
        return None

    gs, trk, vr = ref[GS], ref[TRK], ref[VR]
    if not (math.isfinite(gs) and math.isfinite(trk)) and len(fixes) >= 2:
        # Velocity from the two nearest fixes.
        a, b = (fixes[-2], fixes[-1]) if i == len(fixes) else (fixes[0], fixes[1])
        span = b[T] - a[T]
        if span > 0.2:
            rate = dt / span
            dlon = ((b[LON] - a[LON] + 540.0) % 360.0) - 180.0
            return (ref[LAT] + rate * (b[LAT] - a[LAT]),
                    ref[LON] + rate * dlon,
                    ref[ALT] + rate * (b[ALT] - a[ALT]))
    lat, lon = ref[LAT], ref[LON]
    if math.isfinite(gs) and math.isfinite(trk):
        lat, lon = destination(lat, lon, trk, gs * dt)
    alt = ref[ALT] + (vr * dt if math.isfinite(vr) else 0.0)
    return lat, lon, alt


class Aircraft:
    __slots__ = ("icao", "fields", "fixes", "last_update", "_pending_since")

    def __init__(self, icao: str, history: int):
        self.icao = icao
        self.fields: dict = {}
        self.fixes: deque = deque(maxlen=history)
        self.last_update = 0.0
        self._pending_since: Optional[float] = None

    @property
    def callsign(self) -> str:
        return self.fields.get("callsign") or self.icao

    def altitude_m(self, source: str, geoid_m: float, ground_alt_m: float) -> Optional[float]:
        f = self.fields
        geom, baro = f.get("altitude_geom_ft"), f.get("altitude_ft")
        if baro == "ground" or f.get("on_ground") is True:
            return ground_alt_m
        # Geometric altitude is GNSS height (treated as ellipsoidal);
        # barometric is pressure altitude ~ MSL, so add the geoid undulation.
        if source in ("geom", "auto") and _finite(geom):
            return geom * FT_TO_M
        if source in ("baro", "auto") and _finite(baro):
            return baro * FT_TO_M + geoid_m
        return None


class AircraftStore:
    """Thread-safe store fed by per-field MQTT messages."""

    POSITION_SETTLE_S = 0.08  # lat/lon arrive as separate messages

    def __init__(self, altitude_source="auto", geoid_undulation_m=0.0,
                 ground_alt_m=0.0, history=900, timeout_s=90.0):
        self.altitude_source = altitude_source
        self.geoid_m = geoid_undulation_m
        self.ground_alt_m = ground_alt_m
        self.history = history
        self.timeout_s = timeout_s
        self._ac: dict[str, Aircraft] = {}
        self._lock = threading.RLock()
        self.messages = 0

    # ── ingest ────────────────────────────────────────────────────────────
    def apply_field(self, icao: str, field: str, text: str, now: Optional[float] = None,
                    retained: bool = False):
        """Retained messages (replayed by the broker on subscribe) fill in
        fields but never create track fixes: their age is unknown, and a
        publisher that died without clearing them would leave ghosts."""
        now = time.time() if now is None else now
        icao = icao.upper()
        with self._lock:
            self.messages += 1
            if text == "":
                ac = self._ac.get(icao)
                if ac:
                    ac.fields.pop(field, None)
                return
            ac = self._ac.get(icao)
            if ac is None:
                ac = self._ac[icao] = Aircraft(icao, self.history)
                ac.last_update = now
            ac.fields[field] = self._parse(field, text)
            if retained:
                return
            ac.last_update = now
            if field in ("lat", "lon"):
                ac._pending_since = now

    @staticmethod
    def _parse(field: str, text: str):
        if field == "altitude_ft":
            if text == "ground":
                return "ground"
            try:
                return float(text)
            except ValueError:
                return None
        if field in NUMERIC:
            try:
                return float(text)
            except ValueError:
                return None
        if field == "on_ground":
            return text == "true"
        return text

    def remove(self, icao: str):
        with self._lock:
            self._ac.pop(icao.upper(), None)

    def tick(self, now: Optional[float] = None):
        """Commit settled lat/lon pairs into track history; drop stale aircraft."""
        now = time.time() if now is None else now
        with self._lock:
            for icao in list(self._ac):
                ac = self._ac[icao]
                if now - ac.last_update > self.timeout_s:
                    del self._ac[icao]
                    continue
                p = ac._pending_since
                if p is not None and now - p >= self.POSITION_SETTLE_S:
                    ac._pending_since = None
                    self._commit(ac, p)

    def _commit(self, ac: Aircraft, t: float):
        f = ac.fields
        lat, lon = f.get("lat"), f.get("lon")
        if not (_finite(lat) and _finite(lon)):
            return
        alt = ac.altitude_m(self.altitude_source, self.geoid_m, self.ground_alt_m)
        if alt is None:
            return
        gs = f.get("ground_speed_kt")
        trk = f.get("track") if _finite(f.get("track")) else f.get("true_heading")
        vr = f.get("vertical_rate_fpm")
        fix = [t, lat, lon, alt,
               gs * KT_TO_MS if _finite(gs) else NAN,
               trk if _finite(trk) else NAN,
               vr * FPM_TO_MS if _finite(vr) else NAN]
        if ac.fixes and ac.fixes[-1][LAT] == lat and ac.fixes[-1][LON] == lon:
            ac.fixes[-1][ALT] = alt  # re-published unchanged position
            return
        ac.fixes.append(fix)

    # ── read ──────────────────────────────────────────────────────────────
    def snapshot(self, since: Optional[float] = None) -> dict:
        """Deep-enough copy for rendering/calibration outside the lock.
        ``since`` drops older fixes (always keeps the last two)."""
        with self._lock:
            out = {}
            for icao, ac in self._ac.items():
                if not ac.fixes:
                    continue
                fx = ac.fixes
                if since is not None:
                    keep = [list(x) for x in fx if x[T] >= since]
                    if len(keep) < 2:
                        keep = [list(x) for x in list(fx)[-2:]]
                else:
                    keep = [list(x) for x in fx]
                out[icao] = {"icao": icao, "callsign": ac.fields.get("callsign") or "",
                             "fields": dict(ac.fields), "fixes": keep}
            return out

    def fixes_for(self, icao: str) -> list:
        with self._lock:
            ac = self._ac.get(icao.upper())
            return [list(x) for x in ac.fixes] if ac else []

    def __len__(self):
        """Aircraft with at least one live position."""
        with self._lock:
            return sum(1 for ac in self._ac.values() if ac.fixes)


class MqttFeed:
    """Subscribes to adsb2mqtt topics and feeds an AircraftStore."""

    def __init__(self, store: AircraftStore, host: str, port: int = 1883,
                 prefix: str = "adsb", username: str = "", password: str = "",
                 tls: bool = False, client_id: str = ""):
        self.store = store
        self.host, self.port = host, int(port)
        self.prefix = prefix.rstrip("/")
        self.username, self.password, self.tls = username, password, tls
        self.client_id = client_id
        self.connected = False
        self.last_error = ""
        self._client = None
        self._stop = threading.Event()
        self._ticker: Optional[threading.Thread] = None
        self.on_disappeared: Optional[Callable[[str], None]] = None

    def start(self):
        import paho.mqtt.client as mqtt

        try:  # paho-mqtt >= 2
            c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=self.client_id)
        except (AttributeError, TypeError):  # paho-mqtt 1.x
            c = mqtt.Client(client_id=self.client_id)
        if self.username:
            c.username_pw_set(self.username, self.password or None)
        if self.tls:
            c.tls_set()
        c.on_connect = self._on_connect
        c.on_disconnect = self._on_disconnect
        c.on_message = self._on_message
        c.reconnect_delay_set(1, 30)
        self._client = c
        c.connect_async(self.host, self.port, keepalive=30)
        c.loop_start()
        self._ticker = threading.Thread(target=self._tick_loop, name="aircraft-tick", daemon=True)
        self._ticker.start()
        log.info("MQTT: connecting to %s:%d (prefix '%s')", self.host, self.port, self.prefix)

    def stop(self):
        self._stop.set()
        if self._client:
            self._client.loop_stop()
            try:
                self._client.disconnect()
            except Exception:
                pass

    def _tick_loop(self):
        while not self._stop.wait(0.05):
            self.store.tick()

    # paho v1 / v2 callback signatures differ; accept both via *args.
    def _on_connect(self, client, userdata, flags, rc, *args):
        ok = (rc == 0) if isinstance(rc, int) else not getattr(rc, "is_failure", False)
        self.connected = ok
        if not ok:
            self.last_error = f"connect failed: {rc}"
            log.warning("MQTT %s", self.last_error)
            return
        self.last_error = ""
        client.subscribe([(f"{self.prefix}/aircraft/+/+", 0),
                          (f"{self.prefix}/events/disappeared", 0)])
        log.info("MQTT: connected, subscribed to %s/aircraft/+/+", self.prefix)

    def _on_disconnect(self, client, userdata, *args):
        self.connected = False
        log.warning("MQTT: disconnected")

    def _on_message(self, client, userdata, msg):
        parts = msg.topic.split("/")
        try:
            text = msg.payload.decode("utf-8", errors="replace")
        except Exception:
            return
        # prefix may itself contain '/', so index from the end.
        if len(parts) >= 4 and parts[-3] == "aircraft":
            self.store.apply_field(parts[-2], parts[-1], text, retained=bool(msg.retain))
        elif parts[-1] == "disappeared" and text:
            import json
            try:
                icao = json.loads(text).get("icao24")
            except (ValueError, AttributeError):
                icao = None
            if icao:
                self.store.remove(icao)
                if self.on_disappeared:
                    self.on_disappeared(icao)
