"""Saving the frame index must survive something else holding the file open.

On Windows, replacing a file fails with "access is denied" while anything has the
target open - even just for reading. Antivirus, the search indexer, a sync agent or
an editor watching the folder all do this for a few milliseconds at a time. It used
to surface as a PermissionError out of the auto-log, killing that render pass, and
it left the half-written temporary file behind:

    PermissionError: [WinError 5] Access is denied:
      '...\\data\\frames\\tmp64eqwp6c.tmp' -> 'data\\frames\\index.json'

Holding the file open here reproduces that exactly, on Windows. Elsewhere the
rename just succeeds, so those tests check the ordering guarantees instead.

Run:  python -m pytest -q      (or: python tests/test_index_writes.py)
"""
import json
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adsb_camera.jsonfile import sweep_temp, write_json
from adsb_camera.logbook import LogStore

WINDOWS = sys.platform == "win32"


def frame(level=60):
    return np.full((60, 80, 3), level, np.uint8)


def temp_files(folder):
    return [f for f in os.listdir(folder) if f.endswith(".tmp")]


def test_write_gets_through_once_the_reader_lets_go():
    """The retry is the point: a hold that ends is a hold that should not lose data."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "index.json")
        write_json(path, [{"id": "one"}])

        import threading
        holder = open(path, "r", encoding="utf-8")
        threading.Timer(0.05, holder.close).start()   # let go mid-retry

        write_json(path, [{"id": "two"}])
        with open(path, encoding="utf-8") as fh:
            assert json.load(fh) == [{"id": "two"}], "the write was lost"
        assert not temp_files(tmp), "a temporary file was left behind"


def test_a_hold_that_never_ends_reports_and_tidies_up():
    if not WINDOWS:
        return
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "index.json")
        write_json(path, ["before"])
        holder = open(path, "r", encoding="utf-8")
        try:
            raised = False
            try:
                write_json(path, ["after"], attempts=2)
            except PermissionError:
                raised = True
            assert raised, "a write that cannot land must say so, not pretend"
            assert not temp_files(tmp), "the temporary file was left behind"
        finally:
            holder.close()


def test_autolog_keeps_running_when_the_index_is_locked():
    """This is what crashed the render loop: add() must not raise."""
    if not WINDOWS:
        return
    with tempfile.TemporaryDirectory() as tmp:
        store = LogStore(os.path.join(tmp, "frames"), max_auto=3)
        store.add(frame(), 1000.0, {}, source="auto")

        holder = open(store._index_path, "r", encoding="utf-8")
        try:
            entry = store.add(frame(), 1001.0, {}, source="auto")
            assert entry["id"], "add() should still report the frame it saved"
            assert os.path.exists(store.frame_path(entry["id"])), "the frame itself was lost"
            assert any(e["id"] == entry["id"] for e in store.list()), \
                "the frame should stay listed so the next save picks it up"
        finally:
            holder.close()

        # Once the hold goes, the next save catches everything up.
        store.add(frame(), 1002.0, {}, source="auto")
        with open(store._index_path, encoding="utf-8") as fh:
            assert len(json.load(fh)) == 3


def test_frames_are_not_deleted_while_the_index_still_lists_them():
    """Otherwise a restart offers frames whose picture is gone."""
    if not WINDOWS:
        return
    with tempfile.TemporaryDirectory() as tmp:
        store = LogStore(os.path.join(tmp, "frames"), max_auto=2)
        ids = [store.add(frame(), 1000.0 + i, {}, source="auto")["id"] for i in range(2)]

        holder = open(store._index_path, "r", encoding="utf-8")
        try:
            assert store.delete(ids[0]) is False, "delete should report that it could not"
            assert os.path.exists(store.frame_path(ids[0])), \
                "the file was deleted although the index still lists it"
            assert any(e["id"] == ids[0] for e in store.list())
            assert store.clear() == 0
            assert os.path.exists(store.frame_path(ids[1])), "clear deleted files anyway"
        finally:
            holder.close()

        assert store.delete(ids[0]) is True
        assert not os.path.exists(store.frame_path(ids[0]))


def test_pruning_does_not_orphan_the_index():
    if not WINDOWS:
        return
    with tempfile.TemporaryDirectory() as tmp:
        store = LogStore(os.path.join(tmp, "frames"), max_auto=2)
        for i in range(2):
            store.add(frame(), 1000.0 + i, {}, source="auto")

        holder = open(store._index_path, "r", encoding="utf-8")
        try:
            store.add(frame(), 1002.0, {}, source="auto")     # would prune the oldest
            listed = {e["id"] for e in store.list()}
            for fid in listed:
                assert os.path.exists(store.frame_path(fid)), f"{fid} listed but not on disk"
        finally:
            holder.close()


def test_saving_a_frame_rewrites_the_index_once():
    """At the cap every auto-logged frame also prunes; that is one write, not two.

    The index is the file every other process wants to read, so each rewrite is
    another chance to collide with one of them.
    """
    from adsb_camera import logbook

    with tempfile.TemporaryDirectory() as tmp:
        store = LogStore(os.path.join(tmp, "frames"), max_auto=2)
        for i in range(2):
            store.add(frame(), 1000.0 + i, {}, source="auto")

        writes = []
        real = logbook.write_json
        logbook.write_json = lambda path, data, **kw: (writes.append(path), real(path, data, **kw))[1]
        try:
            store.add(frame(), 1002.0, {}, source="auto")      # adds and prunes
            assert len(writes) == 1, f"the index was rewritten {len(writes)} times"
        finally:
            logbook.write_json = real

        assert len(store.list()) == 2, "the cap should still be enforced"


def test_a_missing_picture_is_not_offered_after_a_restart():
    with tempfile.TemporaryDirectory() as tmp:
        folder = os.path.join(tmp, "frames")
        store = LogStore(folder, max_auto=9)
        keep = store.add(frame(), 1000.0, {}, source="manual")["id"]
        gone = store.add(frame(), 1001.0, {}, source="manual")["id"]
        os.remove(store.frame_path(gone))            # as a half-finished delete would leave it

        again = LogStore(folder, max_auto=9)
        ids = [e["id"] for e in again.list()]
        assert keep in ids and gone not in ids, ids


def test_leftover_temporary_files_are_cleaned_up():
    with tempfile.TemporaryDirectory() as tmp:
        stale = os.path.join(tmp, "tmp64eqwp6c.tmp")
        with open(stale, "w", encoding="utf-8") as fh:
            fh.write("half a write")
        os.utime(stale, (time.time() - 3600, time.time() - 3600))
        fresh = os.path.join(tmp, "tmpfresh.tmp")
        with open(fresh, "w", encoding="utf-8") as fh:
            fh.write("a write happening right now")

        assert sweep_temp(tmp) == 1
        assert not os.path.exists(stale), "the leftover was not cleaned up"
        assert os.path.exists(fresh), "a write in progress must be left alone"


def test_a_new_logstore_sweeps_its_folder():
    with tempfile.TemporaryDirectory() as tmp:
        folder = os.path.join(tmp, "frames")
        os.makedirs(folder)
        stale = os.path.join(folder, "tmpold.tmp")
        with open(stale, "w", encoding="utf-8") as fh:
            fh.write("x")
        os.utime(stale, (time.time() - 3600, time.time() - 3600))

        LogStore(folder)
        assert not os.path.exists(stale)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
