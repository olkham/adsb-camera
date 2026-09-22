"""Small JSON files that must never be left half-written.

Write to a temporary file in the same folder and rename it over the target, so a
reader only ever sees the whole old file or the whole new one.

On Windows that rename fails with "access is denied" if *anything* holds the
target open, even just for reading: antivirus, the search indexer, a backup or
cloud-sync agent, or an editor watching the folder. The holds last milliseconds,
so a short retry gets through. Without one the write is simply lost and the
temporary file is left behind.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time

log = logging.getLogger("adsb_camera.jsonfile")

ATTEMPTS = 5
BACKOFF_S = 0.04


def write_json(path: str, data, attempts: int = ATTEMPTS) -> None:
    """Replace ``path`` with ``data`` as JSON, atomically. Raises OSError if it can't."""
    folder = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=folder, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1)
        for i in range(attempts):
            try:
                os.replace(tmp, path)
                if i:
                    log.info("Wrote %s on attempt %d (something had it open)",
                             os.path.basename(path), i + 1)
                return
            except PermissionError:
                if i == attempts - 1:
                    raise
                time.sleep(BACKOFF_S * (i + 1))
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def sweep_temp(folder: str, min_age_s: float = 60.0) -> int:
    """Delete leftover temporary files from writes that never completed."""
    n = 0
    try:
        names = os.listdir(folder)
    except OSError:
        return 0
    for name in names:
        if not name.endswith(".tmp"):
            continue
        p = os.path.join(folder, name)
        try:
            if time.time() - os.path.getmtime(p) < min_age_s:
                continue        # possibly a write in progress right now
            os.remove(p)
            n += 1
        except OSError:
            pass
    if n:
        log.info("Cleaned up %d leftover temporary file(s) in %s", n, folder)
    return n
