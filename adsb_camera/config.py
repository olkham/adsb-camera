"""Configuration: built-in defaults <- config.yaml <- environment variables.

Any setting can be overridden from the environment (handy for Docker) with
ADSBCAM__<SECTION>__<KEY>, e.g.

    ADSBCAM__MQTT__HOST=192.168.1.241
    ADSBCAM__VIDEO__SOURCE=rtsp://192.168.1.50:8554/video_main
    ADSBCAM__SITE__LAT=51.47

Values are parsed as YAML scalars, so numbers/booleans work as expected.
"""
from __future__ import annotations

import copy
import logging
import os

import yaml

log = logging.getLogger("adsb_camera.config")

DEFAULTS: dict = {
    "video": {
        "source": "rtsp://127.0.0.1:8554/video_main",
        "rtsp_transport": "tcp",
        "reconnect_s": 3.0,
        "width": None,          # only used for local webcams (index sources)
        "height": None,
        "fps": 30,              # requested capture rate; 0 leaves the camera alone
        "fourcc": "MJPG",       # webcams need this to run at speed; "" keeps their default
        "backend": "auto",      # auto, dshow, msmf, v4l2, avfoundation, gstreamer, any
        "exposure_check_s": 3600,   # look for a stuck auto exposure this often; 0 = never
        "dark_level": 40,       # average brightness (0-255) counted as "gone dark"
    },
    "mqtt": {
        "host": "192.168.1.241",
        "port": 1883,
        "prefix": "adsb",
        "username": "",
        "password": "",
        "tls": False,
    },
    "site": {
        "lat": None,
        "lon": None,
        "alt_m": 0.0,            # camera height above mean sea level
        "geoid_m": 47.0,         # geoid undulation (MSL -> ellipsoid), ~47 m in SE England
    },
    "aircraft": {
        "altitude_source": "auto",   # auto (geometric, else baro) | geom | baro
        "max_range_km": 80.0,
        "stale_s": 30.0,
        "trail_s": 60.0,
        "timeout_s": 90.0,
    },
    "camera": {                  # starting guess, used until calibrated
        "model": "pinhole",      # pinhole | fisheye
        "hfov_deg": 70.0,
        "yaw_deg": 0.0,
        "pitch_deg": 10.0,
        "roll_deg": 0.0,
    },
    "overlay": {
        "labels": True,
        "trails": True,
        "grid": True,
        "offscreen": True,
        "points": False,        # always shown while frozen
        "hud": True,
        "reticle": True,        # marker drawn on each in-view aircraft
        "reticle_shape": "circle",   # circle | square | cross | triangle
        "reticle_size": 1.0,    # multiplier on the marker's size, 0.3 - 4.0
    },
    "autodetect": {
        "enabled": False,
        "interval_s": 10.0,
        "search_px": 45,
        "max_points": 200,
        "resolve_every_s": 60.0,
    },
    "logbook": {          # save frames for later calibration, manually or automatically
        "enabled": False,    # auto-log frames around FOV-entry events
        "interval_s": 1.0,
        "window_s": 5.0,     # +/- this many seconds around a track entering the FOV
        "max_auto_frames": 300,
    },
    "server": {
        "host": "0.0.0.0",
        "port": 8090,
        "output_fps": 15,
        "output_max_width": 1920,
        "jpeg_quality": 80,
    },
    "data_dir": "data",
    "log_level": "INFO",
}

ENV_PREFIX = "ADSBCAM__"


def _merge(base: dict, over: dict) -> dict:
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            _merge(cfg, yaml.safe_load(fh) or {})
        log.info("Loaded config from %s", path)
    elif path:
        log.warning("Config file %s not found - using defaults", path)

    for key, raw in os.environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        parts = [p.lower() for p in key[len(ENV_PREFIX):].split("__") if p]
        if not parts:
            continue
        try:
            value = yaml.safe_load(raw) if raw != "" else ""
        except yaml.YAMLError:
            value = raw
        node = cfg
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return cfg
