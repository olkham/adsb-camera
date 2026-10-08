# ADS-B Camera

Overlays live ADS-B traffic from the `adsb2mqtt` broker onto a camera stream
(IPyCam, any RTSP/MJPEG URL, or a local webcam), with **self-calibration**:
the camera's pose, intrinsics and lens distortion don't need to be known.
Aircraft are the calibration targets.

- Annotated MJPEG stream at `/stream.mjpeg` (open in VLC, Home Assistant, a browser…)
- Web UI at `http://<host>:8090` for calibration, overlay options and a sky plot
- Works with normal webcams and 180°+ fisheyes (e.g. the Ximea)

---

## How the calibration works

You know where the camera is. ADS-B tells you where each aircraft is. So every
aircraft you can see is a known **direction** in the sky. Freeze a frame, click
an aircraft, say which one it is, and that's one correspondence
*pixel ↔ direction*. A robust least-squares fit then finds:

| Parameter | Unlocked at |
|---|---|
| Orientation (yaw / pitch / roll) | 2 points |
| Focal length (field of view) | 3 points |
| Radial distortion k1 | 6 points |
| Principal point | 8 points |
| Video ↔ ADS-B sync offset | 8 points on moving aircraft |
| Distortion k2 (k3 for fisheye) | 10 (14) points |
| Camera position correction (lat / lon / height) | 15 points, opt-in |

Any of these can be **locked** with the padlock beside it in the Camera and
Camera location panels: the solver then leaves that value exactly as it is and
fits the rest. The camera location starts locked. To trust a phone's lat/lon
but let the solver correct the height, unlock just the height. With 30
simulated points, a height given 150 m too high comes back within 5 m, and
lat/lon aren't touched. Locking some of yaw/pitch/roll (but not all three) fits
the free angles on their own. Yaw and roll can't be told apart when the camera
points straight up, so don't lock one of them there.

The **sync offset** matters: ADS-B positions arrive 1–3 s late (readsb →
`adsb2mqtt` publishes every 2 s) and video has its own latency. An airliner
moves ~250 m/s, so a 2 s error is several degrees. The solver fits this offset
alongside the lens, and positions are interpolated/dead-reckoned to each
frame's timestamp.

Lens model: *Standard* (pinhole + Brown distortion) for webcams, *Fisheye*
(Kannala-Brandt, as in OpenCV's fisheye module) for wide lenses. In
"Pick automatically" mode it switches to fisheye only when the view is wider
than ~110° or fisheye fits clearly better.

In testing against the simulator (below) 20–25 clicks recover the camera to
~1–2 px across the whole frame, for both a 66° webcam and a zenith-pointing
185° fisheye, starting from a deliberately wrong guess.

---

## Install and run

### Windows

```bat
install.bat
run.bat
```

### Linux / macOS / Raspberry Pi

```bash
./install.sh
./run.sh
```

The installer looks for `python` and `python3` (3.9+). If both exist with
different versions it asks which one to build `.venv` with. It then installs
`requirements.txt`, creates `config.yaml` from `config.example.yaml`, and
creates `data/`. Use `-y` to recreate an existing `.venv` without asking
(`install.sh --python /path/to/python` to force an interpreter).

`run.sh` / `run.bat` pass extra arguments through:

```bash
./run.sh --source http://192.168.1.50:8080/stream.mjpeg --port 8091
```

### Docker

```bash
docker compose up -d
```

Then open `http://<host>:8090`. Defaults are in `docker-compose.yml`; to change
them, copy `.env.example` to `.env` (the install scripts do this) and edit:

```ini
VIDEO_SOURCE=rtsp://192.168.1.50:8554/video_main
MQTT_HOST=192.168.1.241
SITE_LAT=51.4700
SITE_LON=-0.4543
```

Calibration is stored in `./data` and survives rebuilds. For IPyCam on the
same machine, `host.docker.internal` is already mapped. For a USB webcam on
Linux, uncomment the `devices:` block and set `VIDEO_SOURCE=0`. (Docker
Desktop on Windows/macOS can't pass USB cameras through; use IPyCam.)

---

## Configure

Edit `config.yaml` (or use environment variables
`ADSBCAM__<SECTION>__<KEY>`, e.g. `ADSBCAM__MQTT__HOST=192.168.1.241`).

**Video source** from IPyCam:

| Stream | URL |
|---|---|
| RTSP main (preferred, lower latency) | `rtsp://<ipycam-host>:8554/video_main` |
| MJPEG (always available) | `http://<ipycam-host>:8080/stream.mjpeg` |

Use the main stream rather than the sub stream: more pixels means more precise
clicks. If you use a different resolution later, the calibration rescales.

**A local webcam** is `video.source: 0` (or 1, 2 …), with `video.width`,
`video.height` and `video.fps` for what to ask it for. `video.fourcc` matters
more than it looks: a USB webcam hands over uncompressed frames unless asked
otherwise, and the bus cannot carry many of those — a Logitech C920 measured
**4.9 fps at 1080p** that way, and **29.5 fps** once asked for `MJPG`, which is
the default. If a camera refuses MJPG, set `fourcc: ""` to take whatever it
offers. The status line shows the format actually negotiated, so a silent
fallback to an uncompressed one is visible rather than just slow.

`video.backend` picks the capture API (`auto`, `dshow`, `msmf`, `v4l2`,
`avfoundation`, `gstreamer`, `any`). Leave it on `auto` unless a camera will not
open — on Windows, `auto` resolves to DirectShow, and MSMF cannot open some
cameras at all.

**Exposure watchdog.** A webcam left running for hours can settle on an exposure
and stop adapting, leaving the picture dark against a bright sky. Every
`video.exposure_check_s` (3600 by default; 0 turns it off) the average brightness
of the picture is compared with `video.dark_level` (40 out of 255). If it has gone
dark, the camera's auto exposure is re-run — switched off and straight back on, so
the driver works it out again — and if that doesn't help within a few seconds the
camera is reconnected. The brightness it is judging on is in `/api/state` under
`video.brightness`, and every step is logged.

Nothing can tell a stuck exposure from a scene that is honestly dark, so after
both have been tried the camera is left alone until the picture brightens by
itself. Otherwise every night would be spent reconnecting. One consequence: each
nightfall costs one auto-exposure nudge and one reconnect, a blip of a second or
two. Raise `dark_level` to catch a milder fault, lower it (or set
`exposure_check_s: 0`) to leave the camera alone.

**Camera location** (`site.lat`, `site.lon`, `site.alt_m`): the camera, not
the antenna. A phone GPS reading is fine; the angular error from a 20 m
position error is negligible for aircraft kilometres away. `alt_m` is height
above sea level. `geoid_m` converts sea level to the WGS-84 ellipsoid that
ADS-B geometric altitude uses (~47 m across southern England). You can set all
of this in the web UI instead; UI values are saved to `data/calibration.json`
and take precedence over the config file.

**Starting guess** (`camera:` section): only used until the first solve. For
the Ximea with the 180° lens, set `model: fisheye`, `hfov_deg: 180`, and
`pitch_deg: 90` if it looks straight up. The Lens menu in the web UI does the
same.

---

## Calibrate

1. Open the web UI and check the camera location is set (the sky plot tells you).
2. Optional but helpful: drag the **Yaw / Pitch / Horizontal FOV** sliders until
   the sky grid roughly matches (e.g. the horizon line on the horizon). The sky
   plot shows the camera's footprint against the traffic.
3. When you can see an aircraft, press **Freeze frame** (or Space).
4. Click the aircraft. Hover to get a 4× magnifier from the full-resolution frame.
5. Pick it from the list. Before calibration the list is sorted by range with
   azimuth/elevation shown, so cross-check with the sky plot or `adsb-mapper`.
   Once two points exist it's sorted by how close each prediction is to your click.
6. With **Solve after each new point** on (the default) the fit updates immediately.
7. Resume, wait for aircraft elsewhere in the view, repeat.

Aircraft often cross the frame faster than you can freeze and click, especially
near the edges. Two ways around that:

- **Save frame**: instead of freezing, saves the current frame and the
  aircraft data behind it to disk. Live streaming carries on. Come back later
  (even after the aircraft has gone), pick it out of the **Saved frames**
  timeline, and click its aircraft exactly as you would a frozen frame.
- **Auto-log frames when aircraft enter view** (under Save frame): watches for a
  track entering the field of view and saves frames roughly once a second for
  five seconds either side of that moment, so a good pass through the frame is
  captured without you needing to catch it live.

The **Saved frames** panel shows every saved frame as a timeline: a strip of
thumbnails running oldest to newest, with a scrubber under it. Drag the
scrubber (or click a thumbnail, or use ← and →) and the main view loads that
frame as you go, so you can scan a whole auto-logged pass in a second or two
and stop on the frames worth annotating. Breaks of more than two minutes show
as a marked gap, so separate sessions stay apart. An amber dot marks frames you
saved by hand; the rest are auto-logged.

**Delete** removes the selected frame. Once you've annotated what you need and
solved, **Clear all** deletes every saved frame and its thumbnail from
`data/frames/` — the calibration points you already added are kept.

Tips:

- **Spread points out**: corners and edges constrain distortion; the centre
  alone won't. Aim for 15–25 points across the frame.
- **Include aircraft at different times** (and ideally different directions of
  travel) so the sync offset is well determined.
- Points marked **check this** don't fit the others: usually a wrong pick or a
  mis-click. Untick or delete them and Solve again.
- A **landmark** with known lat/lon/height (a mast, a church spire) can be
  added from the picker. Useful at night or for the first point.
- If the camera gets knocked, tick **Keep lens fixed, re-aim only** in Solver
  (it locks the field of view, lens centre and distortion), clear the old
  points, and add 2–3 new ones; the lens model is kept.
- **Add points automatically (experimental)**: once RMS is under ~10 px, the
  app looks for dark (day) or bright (night) blobs near each prediction and
  adds a point when the offset is stable while the aircraft moves. Detections
  far from a good prediction are ignored, and auto points the fit rejects are
  dropped on the next solve. Clear them with **Clear auto points**.

---

## Live tracking

ADS-B positions arrive about once a second, and each one is late by a
*varying* amount (readsb batches, adsb2mqtt publishes on a timer, MQTT
queues). The overlay has to guess where each aircraft is in between. If it
dead-reckons from one fix and snaps to the next, the marker jumps back every
time a fix turns out more delayed than the one before. The **Tracking** panel
controls how the live view fills in:

- **Kalman filter** (default): a constant-velocity filter per aircraft, fed
  each position and the aircraft's own reported speed, heading and climb
  rate. The timing jitter shows up as position noise, which the filter
  averages against the much steadier reported velocity.
- **Straight line between fixes**: the previous behaviour, and what
  calibration uses.
- **Ease corrections over** *n* s (for either one): when a new fix changes
  the estimate, the marker glides to the new track over about that long
  instead of jumping.

In `tests/test_tracker.py` (230 m/s, fixes 0.1–0.9 s late), the straight-line
estimator jumps back up to ~150 m and averages 53 m off. The Kalman filter
never steps back and averages 12 m off, and it stays ahead through a
standard-rate turn too (26 m against 51 m). Frozen and saved frames always
interpolate between the fixes either side, because once those are known it's
the better estimate.

**Picture-in-picture** (toggle under the picture; zoom, count and corner in
Overlay) adds magnified insets of the nearest aircraft in view, cut from the
full-resolution frame, so they have more detail than the stream they sit in.
They are hidden while a frame is frozen, so a click always lands on the real
image.

---

IPyCam's timestamp overlay doesn't interfere with calibration, but you may
prefer to turn it off in IPyCam's Display settings since this app adds its own
status line.

---

## Try it without hardware

`tools/simulate.py` publishes fake aircraft in the `adsb2mqtt` format and
serves a synthetic sky camera with a hidden pose and lens, including a 1 s
ADS-B delay for the solver to discover. It uses the topic prefix `adsb-sim` so
your real feed isn't touched.

```bash
# terminal 1 (needs an MQTT broker; your 192.168.1.241 one is fine)
.venv/bin/python tools/simulate.py --broker 192.168.1.241
#   add --fisheye for a 185° zenith-pointing fisheye
# terminal 2
./run.sh --source http://127.0.0.1:8765/stream.mjpeg
```

Set `mqtt.prefix: adsb-sim` and the camera location to the simulator's
(printed on start, default 51.47, -0.4543, 30 m) in a separate config file or
the UI. Ground truth is at `http://127.0.0.1:8765/camera.json` and
`/truth.json?t=<unix time>`.

Unit tests (synthetic recovery of webcam and fisheye cameras, outlier
detection, projection round-trips):

```bash
.venv/bin/python tests/test_calibration.py     # or: python -m pytest -q
```

---

## HTTP API

| Method | Path | Purpose |
|---|---|---|
| GET | `/stream.mjpeg` | Annotated live stream |
| GET | `/snapshot.jpg` | Latest annotated frame |
| GET | `/api/state` | Camera, aircraft (with pixel positions), points, status |
| POST | `/api/freeze`, `/api/unfreeze` | Freeze / resume |
| GET | `/frozen.jpg` | Frozen frame, full resolution, no overlay |
| POST | `/api/frames` | Save the current frame (+ aircraft data) for later |
| GET | `/api/frames` | List saved frames |
| GET | `/api/frames/{id}/image.jpg` | A saved frame, full resolution, no overlay |
| GET | `/api/frames/{id}/thumb.jpg` | Small preview of a saved frame, for the timeline |
| POST | `/api/frames/{id}/view` | Load a saved frame for calibration (like freezing) |
| DELETE | `/api/frames/{id}` | Delete a saved frame |
| DELETE | `/api/frames` | Delete every saved frame |
| POST | `/api/candidates` `{x, y}` | Aircraft that could be at a normalised click |
| POST | `/api/points` `{x, y, icao}` or `{x, y, landmark:{lat,lon,alt_m,label}}` | Add a point |
| PATCH/DELETE | `/api/points/{id}` | Enable/disable, delete |
| POST | `/api/solve` | Fit now |
| POST | `/api/camera` `{yaw, pitch, roll, hfov, model, sync_offset_s}` | Manual adjust |
| POST | `/api/site`, `/api/settings` | Location; overlay, solver (incl. `solve.locks`) and `tracking` options |
| GET | `/api/calibration` | Download `calibration.json` |

Interactive docs at `/api/docs`. `/api/state` gives every aircraft's `x, y`
(0–1) in the image, so other tools can reuse the calibration.

---

## Accuracy notes

- Positions come from MQTT receipt time plus the fitted sync offset, so a
  steady latency is handled. Variable latency isn't: a stream that buffers
  under CPU load will make labels lag or lead. RTSP over TCP from IPyCam is
  usually steady.
- MLAT and TIS-B positions are noisier than ADS-B; prefer ADS-B aircraft for
  calibration points.
- If an aircraft has no geometric altitude, barometric altitude is used, which
  can be off by a few hundred feet on non-standard-pressure days (a fraction
  of a degree at typical ranges).
- Retained MQTT messages fill in callsigns etc. on startup, but an aircraft is
  only drawn once a live position arrives, so aircraft left behind by a
  crashed publisher never appear as frozen ghosts.

---

## Layout

```
adsb_camera/
  aircraft.py      MQTT feed, per-aircraft track history, interpolation
  camera_model.py  pinhole / fisheye projection, orientation
  calibration.py   correspondences, staged robust solver, persistence
  scene.py         aircraft → pixels, sky grid, FOV outline
  tracker.py       live position estimates: Kalman filter, correction easing
  overlay.py       OpenCV drawing
  autodetect.py    experimental automatic point collection
  logbook.py       saved frames for later calibration (manual + auto-log)
  app.py           orchestration, render loop
  server.py        FastAPI: UI, MJPEG, JSON API
  static/          web UI
tools/simulate.py  fake ADS-B + synthetic camera with ground truth
tests/             synthetic calibration tests
```
