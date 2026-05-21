"""Resolve the active model directory with persistent-volume support.

The deployment baseline (`<repo>/models/`) ships with models baked into the
container image by the GitHub Actions daily retrain. Manual retrains via
`/api/retrain/trigger` previously wrote there too, but Railway containers
have ephemeral filesystems and any in-container writes get wiped on the
next container restart.

This module supports an optional Railway volume mounted at a separate path
(controlled by the `NBA_BETS_MODEL_DIR` env var). When set:
  - Writers (training scripts) save to that path so retrains persist.
  - Readers (API loaders) look there first, then fall back to the image
    baseline if a specific file isn't present in the volume.
  - On first use, the volume gets seeded with whatever is in the image
    baseline so the API doesn't crash on a freshly-mounted empty volume.

Used together with a Railway volume mounted at, e.g., `/app/models_persist`,
this lets manual retrains accumulate state across container restarts without
risking the image's known-good fallback models.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

_log = logging.getLogger(__name__)

# Repo-relative default — matches the image's /app/models/ on Railway.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_IMAGE_DEFAULT_DIR = _REPO_ROOT / "models"

# Track whether we've already attempted seeding to avoid re-running it on
# every call. Once flipped, callers see the volume as it is.
_SEEDED = False


def _is_essentially_empty(p: Path) -> bool:
    """An "empty" model dir has no .pkl files at the top level."""
    if not p.exists():
        return True
    return not any(p.glob("*.pkl"))


def _seed_volume(src: Path, dst: Path) -> None:
    """Copy everything from src into dst that isn't already present.

    Runs at most once per process. Idempotent: if dst already has files
    with matching names they are not overwritten. Walks subdirectories
    (e.g., calibration/) so the JSON/pkl artifacts under them come along.
    """
    if not src.exists():
        _log.warning("Cannot seed volume — source %s missing", src)
        return
    n_copied = 0
    for entry in src.iterdir():
        target = dst / entry.name
        if target.exists():
            continue
        try:
            if entry.is_dir():
                shutil.copytree(entry, target)
            else:
                shutil.copy2(entry, target)
            n_copied += 1
        except Exception as exc:  # noqa: BLE001 — never crash on seed
            _log.warning("Seed copy %s -> %s failed: %s", entry, target, exc)
    if n_copied:
        _log.info("Seeded model volume %s from %s (%d entries)", dst, src, n_copied)


def get_model_dir() -> Path:
    """Return the directory where models should be saved AND loaded.

    Resolution order:
      1. NBA_BETS_MODEL_DIR env var — when set, this is the active dir.
         Seeded from the image baseline on first call if empty.
      2. Repo-default `<repo>/models/` (image baseline).
    """
    global _SEEDED
    override = os.environ.get("NBA_BETS_MODEL_DIR")
    if not override:
        return _IMAGE_DEFAULT_DIR

    active = Path(override)
    try:
        active.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        _log.warning(
            "Could not create NBA_BETS_MODEL_DIR=%s (%s) — using image default",
            active, exc,
        )
        return _IMAGE_DEFAULT_DIR

    if not _SEEDED and active != _IMAGE_DEFAULT_DIR and _is_essentially_empty(active):
        _seed_volume(_IMAGE_DEFAULT_DIR, active)
    _SEEDED = True
    return active


def get_image_default_dir() -> Path:
    """Return the image-baked baseline dir, unconditionally. Used by code that
    needs to read fallback artifacts when the active dir is missing a file."""
    return _IMAGE_DEFAULT_DIR


def resolve_model_file(filename: str) -> Path:
    """Return the Path to a model file, preferring the active dir but falling
    back to the image baseline if the active dir doesn't have it.

    For reading only — writers should use get_model_dir() and stage their
    output there directly so future restarts persist them.
    """
    active = get_model_dir() / filename
    if active.exists():
        return active
    fallback = _IMAGE_DEFAULT_DIR / filename
    return fallback if fallback.exists() else active


__all__ = ['get_model_dir', 'get_image_default_dir', 'resolve_model_file']
