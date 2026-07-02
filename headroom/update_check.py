"""Manual release lookup helpers for ``headroom update``."""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
import importlib.metadata as md
from pathlib import Path
from typing import Any

from packaging.version import InvalidVersion, Version

from headroom.paths import workspace_dir

logger = logging.getLogger(__name__)

PACKAGE_NAME = "headroom-ai"
_PYPI_JSON_URL = f"https://pypi.org/pypi/{PACKAGE_NAME}/json"
_CACHE_FILE = "update_check.json"

# Probe PyPI at most once per day.
_CHECK_TTL_SECONDS = 86_400

_OFF_VALUES = frozenset(("off", "false", "0", "no", "disable", "disabled"))
_TRUE_VALUES = frozenset(("on", "true", "1", "yes", "enable", "enabled"))


def _env_off(name: str, default: str = "on") -> bool:
    """Return True when env var ``name`` is set to a falsey/off value."""
    return os.environ.get(name, default).strip().lower() in _OFF_VALUES


def _env_on(name: str) -> bool:
    """Return True when env var ``name`` is set to a truthy/on value."""
    return os.environ.get(name, "").strip().lower() in _TRUE_VALUES


def is_update_check_enabled() -> bool:
    """Whether the update check / banner should run at all.

    Disabled by ``HEADROOM_UPDATE_CHECK=off``, offline mode
    (``HEADROOM_OFFLINE``), stateless mode
    (``HEADROOM_STATELESS=true``/``1``/``yes``/``on``, matching the proxy's own
    parsing), or any CI environment (``CI`` set).
    """
    from headroom.offline import is_offline

    if is_offline():
        return False
    if _env_off("HEADROOM_UPDATE_CHECK"):
        return False
    if _env_on("HEADROOM_STATELESS"):
        return False
    if os.environ.get("CI", "").strip():
        return False
    return True


def _is_source_checkout() -> bool:
    """True when running from a git checkout (developers manage their tree)."""
    try:
        from headroom._version import _source_root

        return _source_root() is not None
    except Exception:
        return False


def _in_docker() -> bool:
    """Best-effort container detection — image rebuilds, not self-update."""
    try:
        return Path("/.dockerenv").exists() or bool(
            os.environ.get("HEADROOM_IN_DOCKER", "").strip()
        )
    except Exception:
        return False


def installed_version() -> str | None:
    """Return the installed Headroom package version, if import metadata exists."""

    try:
        return md.version(PACKAGE_NAME)
    except md.PackageNotFoundError:
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
    if latest_raw is not None:
        return latest_raw

    info = data.get("info")
    if not isinstance(info, dict):
        return None
    raw = info.get("version")
    if not isinstance(raw, str):
        return None
    try:
        parsed = Version(raw)
    except InvalidVersion:
        return None
    if parsed.is_prerelease and not allow_pre:
        return None
    return raw


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
