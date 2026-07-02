"""Manual release lookup helpers for ``headroom update``."""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from packaging.version import InvalidVersion, Version

from headroom.paths import workspace_dir

logger = logging.getLogger(__name__)

PACKAGE_NAME = "headroom-ai"
_PYPI_JSON_URL = f"https://pypi.org/pypi/{PACKAGE_NAME}/json"
_CACHE_FILE = "update_check.json"


def installed_version() -> str | None:
    """Return the installed Headroom package version, if import metadata exists."""

    try:
        return version(PACKAGE_NAME)
    except PackageNotFoundError:
        return None


def _cache_path() -> Path:
    return workspace_dir() / _CACHE_FILE


def write_cache(latest_version: str, *, now: float | None = None) -> None:
    """Persist manual update lookup metadata for diagnostics."""

    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "last_check": now if now is not None else time.time(),
            "latest_version": latest_version,
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(path)
    except OSError:
        logger.debug("update_check: failed to write cache", exc_info=True)


def _select_latest(data: dict[str, Any], *, allow_pre: bool) -> str | None:
    releases = data.get("releases")
    if not isinstance(releases, dict):
        return None

    latest: Version | None = None
    latest_raw: str | None = None
    for raw, files in releases.items():
        if (
            isinstance(files, list)
            and files
            and all(isinstance(file, dict) and file.get("yanked") for file in files)
        ):
            continue
        try:
            parsed = Version(raw)
        except InvalidVersion:
            continue
        if parsed.is_prerelease and not allow_pre:
            continue
        if latest is None or parsed > latest:
            latest = parsed
            latest_raw = raw
    return latest_raw


def fetch_latest_version(*, allow_pre: bool = False, timeout: float = 4.0) -> str | None:
    """Query the PyPI JSON API for the latest release."""

    req = urllib.request.Request(
        _PYPI_JSON_URL,
        headers={"Accept": "application/json", "User-Agent": "headroom-update"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        logger.debug("update_check: PyPI fetch failed", exc_info=True)
        return None

    return _select_latest(data, allow_pre=allow_pre)


__all__ = [
    "PACKAGE_NAME",
    "fetch_latest_version",
    "installed_version",
    "write_cache",
]
