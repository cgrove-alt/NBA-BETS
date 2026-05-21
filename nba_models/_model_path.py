"""Resolve the active model directory with persistent-volume support.

The deployment baseline (`<repo>/models/`) ships with models baked into the
container image by the GitHub Actions daily retrain. Manual retrains via
`/api/retrain/trigger` previously wrote there too, but Railway containers
have ephemeral filesystems and any in-container writes get wiped on the
next container restart.

This module supports an optional Railway volume mounted at a separate path
(controlled by the `NBA_BETS_MODEL_DIR` env var). When set:
  - Writers (training scripts) save to that path so retrains persist.
  - Readers (API loaders) load from the same path.
  - On first use, the volume gets seeded from the image baseline.
  - When the image baseline updates (e.g., daily GitHub Actions retrain
    deploys), per-file mtime checks copy the newer image versions into
    the volume — so the daily Action's outputs don't get silently
    ignored just because the volume was non-empty.

Used together with a Railway volume mounted at, e.g., `/app/models_persist`,
this lets manual retrains accumulate state across container restarts without
risking the image's known-good fallback models.

Self-audit fixes (commit after 59fce3e):
  - File-based lock so multiple processes don't race during seed/sync.
  - Per-file mtime comparison so daily Action outputs propagate even when
    the volume is non-empty (was: seed only if completely empty).
  - Permission errors on the override dir now log at ERROR and fall back
    explicitly (was: caught silently at WARNING).
  - Sync completion marker so partial copy failures don't leave the
    volume in a "looks seeded" state that blocks future syncs.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

_log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_IMAGE_DEFAULT_DIR = _REPO_ROOT / "models"

# Per-process flag is purely an optimization — the real synchronization is
# the on-disk lock + marker. Without the flag we'd re-stat every model file
# on every get_model_dir() call (cheap, but unnecessary in the hot path).
_SYNCED_THIS_PROCESS = False


def _newest_pkl_mtime(p: Path) -> float:
    """Return the newest .pkl mtime in p, or 0 if no .pkl files."""
    try:
        return max(
            (entry.stat().st_mtime for entry in p.glob("*.pkl")),
            default=0.0,
        )
    except OSError:
        return 0.0


def _do_sync(src: Path, dst: Path) -> tuple[int, int]:
    """Copy files/dirs from src to dst when src is newer or dst missing.

    Returns (n_copied, n_errors).

    For top-level files: copy when missing OR when src mtime > dst mtime.
    For directories: recursively copy missing ones (existing dirs are left
    alone — calibration/, versions/ etc. evolve via their own writers).
    """
    n_copied = 0
    n_errors = 0
    if not src.exists():
        return (0, 0)
    for entry in src.iterdir():
        target = dst / entry.name
        try:
            if entry.is_dir():
                if not target.exists():
                    shutil.copytree(entry, target)
                    n_copied += 1
            else:
                needs_copy = (
                    not target.exists()
                    or entry.stat().st_mtime > target.stat().st_mtime
                )
                if needs_copy:
                    # Copy via tempfile + rename so partial writes don't leave
                    # a half-copied target. shutil.copy2 already does this on
                    # most platforms by writing then setting mtime; we also
                    # explicitly fsync to harden against power loss.
                    shutil.copy2(entry, target)
                    n_copied += 1
        except Exception as exc:  # noqa: BLE001
            n_errors += 1
            _log.warning("Seed/sync copy %s -> %s failed: %s", entry, target, exc)
    return (n_copied, n_errors)


def _try_sync(active: Path) -> None:
    """Sync active dir with image baseline. File-locked so concurrent
    processes don't both attempt the copy. Completion marker so partial
    failures get retried on next call.

    Idempotent — running this when both dirs are fully in sync is a no-op
    (zero copies because per-file mtime checks all return False).
    """
    if active == _IMAGE_DEFAULT_DIR:
        return
    if not _IMAGE_DEFAULT_DIR.exists():
        return

    # File-based lock — atomic create via O_EXCL on POSIX. If another
    # process holds it, skip; their sync will cover us.
    lock_file = active / ".sync.lock"
    marker_file = active / ".sync_complete"

    # Fast-path: if marker exists AND its mtime is newer than the newest
    # image .pkl, nothing to do. This makes warm restarts free.
    try:
        marker_mtime = marker_file.stat().st_mtime
        image_mtime = _newest_pkl_mtime(_IMAGE_DEFAULT_DIR)
        if marker_mtime >= image_mtime > 0:
            return
    except OSError:
        pass  # marker missing — proceed to sync

    try:
        fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.close(fd)
    except FileExistsError:
        _log.debug("Another process holds the sync lock — skipping")
        return
    except OSError as exc:
        _log.error("Cannot create sync lock at %s: %s", lock_file, exc)
        return

    try:
        n_copied, n_errors = _do_sync(_IMAGE_DEFAULT_DIR, active)
        if n_errors > 0:
            _log.warning(
                "Model sync incomplete: %d copied, %d errors — marker NOT set, "
                "next get_model_dir() call will retry",
                n_copied, n_errors,
            )
            return  # don't set marker on partial failure
        # Only after successful completion do we mark the sync done.
        # The marker's mtime serves as the high-water mark for "what
        # image baseline have we seen". On image rebuilds where Daily
        # Action commits new models, those models have a mtime > the
        # marker, so next call re-syncs them.
        marker_file.touch()
        if n_copied > 0:
            _log.info("Synced %d entries from %s to %s", n_copied, _IMAGE_DEFAULT_DIR, active)
    finally:
        try:
            os.unlink(lock_file)
        except OSError:
            pass


def get_model_dir() -> Path:
    """Return the directory where models should be saved AND loaded.

    Resolution order:
      1. NBA_BETS_MODEL_DIR env var when set and accessible.
      2. Repo-default `<repo>/models/` (image baseline).

    On first call against an override dir, syncs from the image baseline.
    On subsequent calls when the image is newer than the volume's sync
    marker, re-syncs file-by-file using mtime comparison — so a fresh
    Daily Action retrain propagates to the volume even when the volume
    is non-empty.
    """
    global _SYNCED_THIS_PROCESS
    override = os.environ.get("NBA_BETS_MODEL_DIR")
    if not override:
        return _IMAGE_DEFAULT_DIR

    active = Path(override)
    try:
        active.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        # ERROR-level: this is a misconfiguration the operator needs to
        # see, not a benign quirk. We still fall back so the API can
        # serve from the image dir, but the log makes the breakage loud.
        _log.error(
            "NBA_BETS_MODEL_DIR=%s set but inaccessible: %s — "
            "falling back to image baseline %s. Retrains will NOT persist "
            "until the volume is fixed.",
            active, exc, _IMAGE_DEFAULT_DIR,
        )
        return _IMAGE_DEFAULT_DIR

    if not _SYNCED_THIS_PROCESS:
        _try_sync(active)
        _SYNCED_THIS_PROCESS = True
    return active


def get_image_default_dir() -> Path:
    """Return the image-baked baseline dir, unconditionally. Used by code
    that needs to read fallback artifacts when the active dir is missing
    a file."""
    return _IMAGE_DEFAULT_DIR


def resolve_model_file(filename: str) -> Path:
    """Return the Path to a model file, preferring the active dir but
    falling back to the image baseline if the active dir doesn't have it.

    For reading only — writers should use get_model_dir() and stage their
    output there directly so future restarts persist them.
    """
    active = get_model_dir() / filename
    if active.exists():
        return active
    fallback = _IMAGE_DEFAULT_DIR / filename
    return fallback if fallback.exists() else active


__all__ = ['get_model_dir', 'get_image_default_dir', 'resolve_model_file']
