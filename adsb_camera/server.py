"""HTTP layer: web UI, annotated MJPEG stream and a small JSON API."""
from __future__ import annotations

import asyncio
import os
from typing import Optional

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .app import AdsbCamera

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
NO_CACHE = {"Cache-Control": "no-store"}


def create_app(core: AdsbCamera) -> FastAPI:
    api = FastAPI(title="ADS-B Camera", docs_url="/api/docs", redoc_url=None)
    api.mount("/static", StaticFiles(directory=STATIC), name="static")

    @api.get("/", include_in_schema=False)
    def index():
        return FileResponse(os.path.join(STATIC, "index.html"), headers=NO_CACHE)

    @api.get("/stream.mjpeg", summary="Annotated live stream (MJPEG)")
    async def stream():
        async def gen():
            core.clients += 1
            last = -1
            try:
                while True:
                    jpg, seq = core.latest_jpeg()
                    if jpg is not None and seq != last:
                        last = seq
                        yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                               + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                    await asyncio.sleep(0.02)
            finally:
                core.clients -= 1

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame",
                                 headers=NO_CACHE)

    @api.get("/snapshot.jpg", summary="Latest annotated frame")
    def snapshot():
        jpg, _ = core.latest_jpeg()
        if jpg is None:
            raise HTTPException(503, "No frame yet")
        return Response(jpg, media_type="image/jpeg", headers=NO_CACHE)

    @api.get("/frozen.jpg", summary="The frozen frame, full resolution, no overlay")
    def frozen():
        jpg = core.frozen_jpeg()
        if jpg is None:
            raise HTTPException(404, "Not frozen")
        return Response(jpg, media_type="image/jpeg", headers=NO_CACHE)

    @api.get("/api/state")
    def state():
        return JSONResponse(core.state(), headers=NO_CACHE)

    @api.post("/api/freeze")
    def freeze():
        return core.freeze()

    @api.post("/api/unfreeze")
    def unfreeze():
        return core.unfreeze()

    @api.post("/api/candidates", summary="Aircraft that could be at a clicked point")
    def candidates(x: float = Body(...), y: float = Body(...)):
        return core.candidates(x, y)

    @api.post("/api/points", summary="Add a calibration point (normalised x/y)")
    def add_point(x: float = Body(...), y: float = Body(...), icao: Optional[str] = Body(None),
                  landmark: Optional[dict] = Body(None)):
        return core.add_point(x, y, icao, landmark)

    @api.patch("/api/points/{pid}")
    def update_point(pid: str, enabled: Optional[bool] = Body(None, embed=True)):
        return core.update_point(pid, enabled)

    @api.delete("/api/points/{pid}")
    def delete_point(pid: str):
        return core.delete_points(pid=pid)

    @api.delete("/api/points")
    def delete_points(source: Optional[str] = None):
        return core.delete_points(source=source)

    @api.post("/api/solve")
    def run_solve(options: Optional[dict] = Body(None, embed=True)):
        return core.run_solve(options)

    @api.post("/api/camera", summary="Manual camera adjustment")
    def set_camera(d: dict = Body(...)):
        return core.set_camera(d)

    @api.post("/api/camera/reset")
    def reset_camera():
        return core.reset_camera()

    @api.post("/api/site")
    def set_site(d: dict = Body(...)):
        return core.set_site(d)

    @api.post("/api/settings")
    def set_settings(d: dict = Body(...)):
        return core.set_settings(d)

    @api.get("/api/calibration", summary="Download calibration.json")
    def download():
        path = core.calib_store.path
        if not os.path.exists(path):
            raise HTTPException(404, "No calibration saved yet")
        return FileResponse(path, media_type="application/json", filename="calibration.json")

    return api
