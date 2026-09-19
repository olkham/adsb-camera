"""A still frame must be sent more than once, or the browser never paints it.

Browsers show a multipart JPEG part only once the next part starts to arrive, so
whatever was sent last is still waiting behind the one before it. Live video hides
this - the next frame is along in a moment - but a frozen frame or a saved frame
being reviewed has nothing behind it, and the viewer keeps showing the previous
picture until something else moves. That is the "have to click it twice" bug, and
it is worst exactly where it matters: picking aircraft out of a saved frame.

Run:  python -m pytest -q      (or: python tests/test_stream_paints.py)
"""
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adsb_camera.server import LIVE_REPEAT_S, STILL_REPEAT_S, mjpeg_stream


class FakeCore:
    """Stands in for AdsbCamera: hands out jpegs the way the render loop would."""

    def __init__(self, frozen, moving):
        self.frozen = {"t": 1.0} if frozen else None
        self.moving = moving           # a new frame every read, as live video does
        self.clients = 0
        self.seq = 1

    def latest_jpeg(self):
        if self.moving:
            self.seq += 1
        return b"\xff\xd8" + str(self.seq).encode(), self.seq


def collect(core, seconds):
    """Parts a viewer receives over this long."""
    async def run():
        out, deadline = [], time.monotonic() + seconds
        gen = mjpeg_stream(core, sleep_s=0.002)
        try:
            while time.monotonic() < deadline:
                out.append(await asyncio.wait_for(gen.__anext__(), timeout=seconds))
        except (asyncio.TimeoutError, StopAsyncIteration):
            pass
        finally:
            await gen.aclose()
        return out
    return asyncio.run(run())


def test_still_frame_is_repeated_so_it_gets_painted():
    core = FakeCore(frozen=True, moving=False)
    parts = collect(core, STILL_REPEAT_S + 0.15)
    assert len(parts) >= 2, f"a frozen frame was sent {len(parts)} time(s); the viewer would not paint it"
    assert parts[0] == parts[1], "the repeat must be the same picture"


def test_still_frame_is_not_repeated_forever():
    """One repeat is enough; re-sending a static picture endlessly is just traffic."""
    core = FakeCore(frozen=True, moving=False)
    parts = collect(core, STILL_REPEAT_S + 0.4)
    assert len(parts) == 2, f"a still frame was sent {len(parts)} times, expected 2"


def test_live_video_is_not_doubled_up():
    core = FakeCore(frozen=False, moving=True)
    parts = collect(core, LIVE_REPEAT_S * 0.6)
    assert len(parts) == len(set(parts)), "live frames were duplicated onto the wire"
    assert len(parts) > 5, "expected a run of live frames"


def test_viewer_count_is_released():
    core = FakeCore(frozen=True, moving=False)
    collect(core, 0.05)
    assert core.clients == 0, "a disconnected viewer is still counted"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
