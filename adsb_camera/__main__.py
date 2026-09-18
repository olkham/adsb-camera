"""python -m adsb_camera [--config config.yaml] [--port 8090] [--source URL]"""
from __future__ import annotations

import argparse
import logging
import sys

from . import __version__
from .config import load_config


def main(argv=None):
    ap = argparse.ArgumentParser(prog="adsb_camera",
                                 description="Overlay live ADS-B aircraft on a camera stream.")
    ap.add_argument("-c", "--config", default="config.yaml", help="YAML config file")
    ap.add_argument("--source", help="override video.source (RTSP/MJPEG URL, file, or camera index)")
    ap.add_argument("--mqtt-host", help="override mqtt.host")
    ap.add_argument("--host", help="override server.host")
    ap.add_argument("--port", type=int, help="override server.port")
    ap.add_argument("--log-level", help="DEBUG, INFO, WARNING ...")
    ap.add_argument("--version", action="version", version=__version__)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.source is not None:
        cfg["video"]["source"] = args.source
    if args.mqtt_host:
        cfg["mqtt"]["host"] = args.mqtt_host
    if args.host:
        cfg["server"]["host"] = args.host
    if args.port:
        cfg["server"]["port"] = args.port
    level = (args.log_level or cfg.get("log_level") or "INFO").upper()
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    import uvicorn

    from .app import AdsbCamera
    from .server import create_app

    core = AdsbCamera(cfg)
    core.start()
    host, port = cfg["server"]["host"], int(cfg["server"]["port"])
    shown = "localhost" if host in ("0.0.0.0", "::") else host
    logging.getLogger("adsb_camera").info("Web UI: http://%s:%d/   stream: /stream.mjpeg", shown, port)
    try:
        uvicorn.run(create_app(core), host=host, port=port, log_level="warning")
    finally:
        core.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
