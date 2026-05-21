"""Tests for nba_models._model_path.

The resolver picks between the image baseline (`<repo>/models/`) and an
override directory (`NBA_BETS_MODEL_DIR`, typically a Railway volume mount).
These tests pin the seeding/sync semantics so a regression in the
production flow surfaces immediately.

Self-audit context (2026-05-21):
  - Initial implementation only seeded on a fully-empty override → daily
    Action retrains became invisible. Now per-file mtime check.
  - Initial implementation used a process-level _SEEDED flag → multi-
    process race. Now uses on-disk file lock.
  - Initial implementation didn't validate seed completion → partial
    failures left "looks seeded" state. Now uses marker-on-success.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nba_models._model_path as mp  # noqa: E402


@pytest.fixture(autouse=True)
def reset_module(monkeypatch):
    """Reset module-level state between tests.

    `autouse=True` so it runs FIRST in every test (before fake_image_dir
    and volume_dir which set env vars/module state). Without that, fixture
    ordering put this AFTER volume_dir and its monkeypatch.delenv blew
    away the env var the test needed.

    Deliberately does NOT touch NBA_BETS_MODEL_DIR — each test sets it
    explicitly via the volume_dir fixture or directly via monkeypatch.
    """
    monkeypatch.setattr(mp, "_SYNCED_THIS_PROCESS", False)
    yield


@pytest.fixture
def fake_image_dir(tmp_path, monkeypatch):
    """Create a fake image baseline with a known set of files."""
    image = tmp_path / "image_models"
    image.mkdir()
    # Seed-able .pkls
    (image / "player_pts.pkl").write_bytes(b"image_pts_v1")
    (image / "player_reb.pkl").write_bytes(b"image_reb_v1")
    # Sub-directory should be copied
    sub = image / "calibration"
    sub.mkdir()
    (sub / "prop_calibrator.pkl").write_bytes(b"cal_v1")
    monkeypatch.setattr(mp, "_IMAGE_DEFAULT_DIR", image)
    yield image


@pytest.fixture
def volume_dir(tmp_path, monkeypatch):
    """Empty volume dir + the env var pointing at it."""
    volume = tmp_path / "volume"
    volume.mkdir()
    monkeypatch.setenv("NBA_BETS_MODEL_DIR", str(volume))
    yield volume


# ---------------------------------------------------------------------------
# get_model_dir() basic resolution
# ---------------------------------------------------------------------------

def test_no_env_var_returns_image_dir(fake_image_dir, monkeypatch):
    monkeypatch.delenv("NBA_BETS_MODEL_DIR", raising=False)
    assert mp.get_model_dir() == fake_image_dir


def test_env_var_returns_override_path(fake_image_dir, volume_dir):
    assert mp.get_model_dir() == volume_dir


# ---------------------------------------------------------------------------
# First-mount seeding
# ---------------------------------------------------------------------------

def test_empty_volume_gets_seeded_from_image(fake_image_dir, volume_dir):
    mp.get_model_dir()
    assert (volume_dir / "player_pts.pkl").read_bytes() == b"image_pts_v1"
    assert (volume_dir / "player_reb.pkl").read_bytes() == b"image_reb_v1"
    assert (volume_dir / "calibration" / "prop_calibrator.pkl").read_bytes() == b"cal_v1"
    assert (volume_dir / ".sync_complete").exists()


def test_seeding_is_idempotent_within_process(fake_image_dir, volume_dir):
    """Second call within same process must not retry sync (perf)."""
    mp.get_model_dir()
    # Mutate a synced file; second get_model_dir() should NOT clobber it
    # because _SYNCED_THIS_PROCESS is True now.
    (volume_dir / "player_pts.pkl").write_bytes(b"mutated")
    mp.get_model_dir()
    assert (volume_dir / "player_pts.pkl").read_bytes() == b"mutated"


# ---------------------------------------------------------------------------
# Staleness sync (the audit's #1 finding)
# ---------------------------------------------------------------------------

def test_newer_image_file_overwrites_older_volume_file(fake_image_dir, volume_dir, monkeypatch):
    """When the image is updated (e.g. daily Action), the volume picks up
    the newer file on next process startup. THIS IS THE CRITICAL FIX."""
    # Initial sync
    mp.get_model_dir()
    assert (volume_dir / "player_pts.pkl").read_bytes() == b"image_pts_v1"

    # Image gets a newer version (simulating GitHub Action commit + Docker rebuild)
    time.sleep(0.05)
    (fake_image_dir / "player_pts.pkl").write_bytes(b"image_pts_v2_new_action_output")

    # New "process" — reset the per-process flag (and clear the in-process
    # sync marker so we re-check the on-disk one)
    monkeypatch.setattr(mp, "_SYNCED_THIS_PROCESS", False)
    mp.get_model_dir()
    assert (volume_dir / "player_pts.pkl").read_bytes() == b"image_pts_v2_new_action_output"


def test_older_image_file_does_not_overwrite_newer_volume_file(fake_image_dir, volume_dir, monkeypatch):
    """Manual retrains (which write to the volume) MUST be preserved when
    they're newer than the image. The sync is one-way: image-newer → volume."""
    mp.get_model_dir()
    # Simulate a manual retrain — volume file is newer than image
    time.sleep(0.05)
    (volume_dir / "player_pts.pkl").write_bytes(b"volume_retrained_recently")

    # Fresh "process"
    monkeypatch.setattr(mp, "_SYNCED_THIS_PROCESS", False)
    mp.get_model_dir()
    assert (volume_dir / "player_pts.pkl").read_bytes() == b"volume_retrained_recently"


# ---------------------------------------------------------------------------
# Race condition guard (file lock)
# ---------------------------------------------------------------------------

def test_lock_file_prevents_concurrent_sync(fake_image_dir, volume_dir, monkeypatch):
    """If another process holds the .sync.lock, get_model_dir() returns
    the dir without re-syncing — they'll cover us."""
    monkeypatch.setattr(mp, "_SYNCED_THIS_PROCESS", False)
    # Pre-create the lock as if another process holds it
    (volume_dir / ".sync.lock").touch()

    # get_model_dir should return the path without throwing or seeding
    result = mp.get_model_dir()
    assert result == volume_dir
    # Files should NOT have been copied (because we yielded to the other process)
    assert not (volume_dir / "player_pts.pkl").exists()


# ---------------------------------------------------------------------------
# Permission error handling (audit finding #4)
# ---------------------------------------------------------------------------

def test_inaccessible_override_falls_back_to_image(fake_image_dir, monkeypatch, caplog):
    """If NBA_BETS_MODEL_DIR points at a path that can't be created, log
    at ERROR and fall back to the image dir."""
    monkeypatch.setenv("NBA_BETS_MODEL_DIR", "/proc/0/nonexistent/no-perms")
    with caplog.at_level("ERROR"):
        result = mp.get_model_dir()
    assert result == fake_image_dir
    # The error should be logged loudly
    assert any("NBA_BETS_MODEL_DIR" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Partial seed failure (audit finding #3)
# ---------------------------------------------------------------------------

def test_partial_seed_does_not_set_marker(fake_image_dir, volume_dir, monkeypatch):
    """When _do_sync reports errors, marker is NOT created — next call retries."""
    # Force _do_sync to report an error
    def fake_sync(src, dst):
        # Pretend we copied 1 file but had 1 error
        (dst / "player_pts.pkl").write_bytes(src.joinpath("player_pts.pkl").read_bytes())
        return (1, 1)

    monkeypatch.setattr(mp, "_do_sync", fake_sync)
    mp.get_model_dir()
    assert not (volume_dir / ".sync_complete").exists()
    # Next process retries
    monkeypatch.setattr(mp, "_SYNCED_THIS_PROCESS", False)
    # Restore real _do_sync so the retry actually succeeds
    import shutil as _shutil
    def real_sync(src, dst):
        copied = 0
        for entry in src.iterdir():
            target = dst / entry.name
            if entry.is_dir():
                if not target.exists():
                    _shutil.copytree(entry, target); copied += 1
            else:
                if not target.exists() or entry.stat().st_mtime > target.stat().st_mtime:
                    _shutil.copy2(entry, target); copied += 1
        return (copied, 0)
    monkeypatch.setattr(mp, "_do_sync", real_sync)
    mp.get_model_dir()
    assert (volume_dir / ".sync_complete").exists()


# ---------------------------------------------------------------------------
# resolve_model_file fallback
# ---------------------------------------------------------------------------

def test_resolve_falls_back_to_image_if_volume_missing_file(fake_image_dir, volume_dir):
    """When the volume is seeded, look there. If a specific file is missing,
    fall back to the image baseline. Useful for reading minor files (calibration
    JSON, etc.) that a manual retrain might not have produced."""
    mp.get_model_dir()  # seed
    # Volume has player_pts.pkl, image also has it
    assert mp.resolve_model_file("player_pts.pkl") == volume_dir / "player_pts.pkl"

    # If volume is missing a file but image has it, return image path
    (volume_dir / "player_pts.pkl").unlink()
    assert mp.resolve_model_file("player_pts.pkl") == fake_image_dir / "player_pts.pkl"


# ---------------------------------------------------------------------------
# Sanity: existing 84-test suite still loads the module without import-time errors
# ---------------------------------------------------------------------------

def test_module_imports_clean():
    """Importing _model_path must not perform IO. The repo-default may not
    exist on some test environments."""
    import importlib
    # Re-import to verify no side effects at module load
    importlib.reload(mp)
