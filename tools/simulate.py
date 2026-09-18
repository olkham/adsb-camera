"""Hardware-free demo / test rig.

* Publishes fake aircraft to MQTT in the adsb2mqtt per-field format.
* Serves an MJPEG "sky camera" at http://<host>:<port>/stream.mjpeg showing
  those aircraft through a hidden camera with a pose/lens you choose.
* Exposes the ground truth at /truth.json?t=<unix time> and /camera.json.

ADS-B positions are published every 2 s and are ``--adsb-delay`` seconds old
when they arrive, like the real pipeline, so the app has to find the sync
offset too.

    python tools/simulate.py --broker 127.0.0.1 --lat 51.47 --lon -0.45
    python -m adsb_camera --source http://127.0.0.1:8765/stream.mjpeg --mqtt-host 127.0.0.1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adsb_camera.camera_model import CameraModel  # noqa: E402
from adsb_camera.geo import FT_TO_M, LocalFrame, az_el_range, destination  # noqa: E402

AIRLINES = ["BAW", "EZY", "RYR", "VIR", "DLH", "AFR", "KLM", "UAE", "SHT", "TOM"]


class Plane:
    def __init__(self, rng, site, yaw, t0):
        self.icao = "%06X" % rng.randrange(0x400000, 0x43FFFF)
        self.callsign = f"{rng.choice(AIRLINES)}{rng.randint(10, 9999)}"
        brg = yaw + rng.uniform(-60, 60)
        self.lat0, self.lon0 = destination(site[0], site[1], brg, rng.uniform(6000, 35000))
        self.trk = rng.uniform(0, 360)
        self.gs = rng.uniform(140, 240)                 # m/s
        self.alt0 = rng.uniform(1500, 11500)            # m (geometric)
        self.vr = rng.choice([0, 0, rng.uniform(-8, 8)])
        self.t0 = t0
        self.life = rng.uniform(120, 400)

    def at(self, t):
        dt = t - self.t0
        lat, lon = destination(self.lat0, self.lon0, self.trk, self.gs * dt)
        return lat, lon, self.alt0 + self.vr * dt


class Sim:
    def __init__(self, a):
        self.a = a
        self.rng = random.Random(a.seed)
        self.site = (a.lat, a.lon, a.alt + a.geoid)
        self.frame = LocalFrame(*self.site)
        model = "fisheye" if a.fisheye else "pinhole"
        w, h = (a.width, a.height) if not a.fisheye else (a.height, a.height)
        self.cam = CameraModel.default(w, h, model, a.hfov, a.yaw, a.pitch, a.roll)
        self.cam.dist = [0.02, -0.01, 0.0, 0.0] if a.fisheye else [-0.12, 0.04, 0.0]
        self.cam.cx += 12
        self.cam.cy -= 8
        now = time.time()
        self.planes = [Plane(self.rng, self.site, a.yaw, now - self.rng.uniform(0, 60)) for _ in range(a.n)]
        self.jpeg = None
        self.lock = threading.Lock()

    def respawn(self, now):
        for i, p in enumerate(self.planes):
            if now - p.t0 > p.life:
                self.planes[i] = Plane(self.rng, self.site, self.a.yaw, now)

    def truth(self, t):
        out = []
        for p in self.planes:
            enu = self.frame.enu(*p.at(t))
            uv, ok = self.cam.project_world(enu)
            _, el, rng = az_el_range(enu)
            if ok[0] and self.cam.in_image(uv)[0] and el > 0:
                out.append({"icao": p.icao, "callsign": p.callsign, "u": float(uv[0][0]),
                            "v": float(uv[0][1]), "range_m": float(rng)})
        return out

    def render(self, t):
        if getattr(self, "_bg", None) is None:
            self._bg = self._background()
        img = self._bg.copy()
        for tr in self.truth(t):
            r = max(1.5, 900.0 / (tr["range_m"] / 1000.0 + 5) / 25)
            cv2.ellipse(img, (int(tr["u"] * 4), int(tr["v"] * 4)), (int(r * 2.2 * 4), int(r * 4)), 0, 0, 360,
                        (45, 45, 55), -1, cv2.LINE_AA, shift=2)
        noise = np.random.default_rng().normal(0, 2.0, img.shape).astype(np.int16)
        return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    def _background(self):
        cam = self.cam
        h, w = cam.height, cam.width
        ys, xs = np.mgrid[0:h:4, 0:w:4]
        rays = cam.unproject(np.stack([xs.ravel(), ys.ravel()], 1).astype(float)) @ cam.R_c2w.T
        el = np.degrees(np.arcsin(np.clip(rays[:, 2], -1, 1))).reshape(xs.shape)
        sky = np.clip(el / 60.0, 0, 1)[..., None]
        img = (np.array([235, 205, 170]) * (1 - sky) + np.array([200, 140, 70]) * sky)
        img[el < 0] = (60, 90, 70)
        back, ok = cam.project_cam(cam.unproject(np.stack([xs.ravel(), ys.ravel()], 1).astype(float)))
        unimaged = (~ok | (np.linalg.norm(back - np.stack([xs.ravel(), ys.ravel()], 1), axis=1) > 1))
        img[unimaged.reshape(xs.shape)] = (0, 0, 0)  # outside the fisheye image circle
        return cv2.resize(img.astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)

    # ── loops ─────────────────────────────────────────────────────────────
    def video_loop(self):
        while True:
            t = time.time()
            ok, buf = cv2.imencode(".jpg", self.render(t), [cv2.IMWRITE_JPEG_QUALITY, 85])
            with self.lock:
                self.jpeg = buf.tobytes()
            time.sleep(max(0.0, 1.0 / self.a.fps - (time.time() - t)))

    def mqtt_loop(self):
        import paho.mqtt.client as mqtt
        try:
            c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="adsb-sim")
        except AttributeError:
            c = mqtt.Client(client_id="adsb-sim")
        c.connect(self.a.broker, self.a.broker_port)
        c.loop_start()
        pre = self.a.prefix
        while True:
            now = time.time()
            self.respawn(now)
            t_pos = now - self.a.adsb_delay
            for p in self.planes:
                lat, lon, alt = p.at(t_pos)
                fields = {"callsign": p.callsign, "lat": f"{lat:.6f}", "lon": f"{lon:.6f}",
                          "altitude_geom_ft": round(alt / FT_TO_M),
                          "altitude_ft": round((alt - self.a.geoid) / FT_TO_M),
                          "ground_speed_kt": round(p.gs / 0.514444, 1), "track": round(p.trk, 1),
                          "vertical_rate_fpm": round(p.vr / 0.00508)}
                for k, v in fields.items():
                    c.publish(f"{pre}/aircraft/{p.icao}/{k}", str(v), retain=True)
            time.sleep(2.0)


def serve(sim: Sim, port: int):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/stream.mjpeg":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                last = None
                try:
                    while True:
                        with sim.lock:
                            j = sim.jpeg
                        if j is not None and j is not last:
                            last = j
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(j)).encode() + b"\r\n\r\n" + j + b"\r\n")
                        time.sleep(0.01)
                except (BrokenPipeError, ConnectionResetError):
                    return
            elif u.path == "/truth.json":
                t = float(parse_qs(u.query).get("t", [time.time()])[0])
                self._json(sim.truth(t))
            elif u.path == "/camera.json":
                self._json(sim.cam.to_dict())
            else:
                self.send_error(404)

        def _json(self, obj):
            b = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--broker", default="127.0.0.1")
    ap.add_argument("--broker-port", type=int, default=1883)
    ap.add_argument("--prefix", default="adsb-sim",
                    help="topic prefix (default adsb-sim so a real feed isn't polluted)")
    ap.add_argument("--lat", type=float, default=51.4700)
    ap.add_argument("--lon", type=float, default=-0.4543)
    ap.add_argument("--alt", type=float, default=30.0, help="camera height above sea level, m")
    ap.add_argument("--geoid", type=float, default=47.0)
    ap.add_argument("--yaw", type=float, default=230.0)
    ap.add_argument("--pitch", type=float, default=12.0)
    ap.add_argument("--roll", type=float, default=1.5)
    ap.add_argument("--hfov", type=float, default=64.0)
    ap.add_argument("--fisheye", action="store_true", help="square 185° fisheye instead of a webcam")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=float, default=10)
    ap.add_argument("--n", type=int, default=14, help="number of aircraft")
    ap.add_argument("--adsb-delay", type=float, default=1.0)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--seed", type=int, default=4)
    a = ap.parse_args()
    if a.fisheye and a.hfov == 64.0:
        a.hfov, a.pitch = 185.0, 85.0
    sim = Sim(a)
    threading.Thread(target=sim.video_loop, daemon=True).start()
    threading.Thread(target=sim.mqtt_loop, daemon=True).start()
    print(f"Simulated camera: http://127.0.0.1:{a.port}/stream.mjpeg  (MQTT prefix '{a.prefix}')")
    print(f"Camera location: lat {a.lat} lon {a.lon} alt {a.alt} m (geoid {a.geoid} m)")
    serve(sim, a.port)


if __name__ == "__main__":
    main()
