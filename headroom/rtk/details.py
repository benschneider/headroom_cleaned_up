"""Detail retrieval for RTK/CCR compression hashes."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import sys


DEFAULT_CCR_DB = Path.home() / ".headroom" / "ccr_store.db"


class DetailLookupError(RuntimeError):
    """Raised when a compressed-detail hash cannot be resolved."""


def default_ccr_db_path() -> Path:
    configured = os.environ.get("HEADROOM_CCR_SQLITE_PATH")
    return Path(configured).expanduser() if configured else DEFAULT_CCR_DB


def retrieve_detail(hash_key: str, db_path: Path | None = None) -> str:
    """Return the stored original content for a CCR/RTK detail hash."""

    normalized = hash_key.strip()
    if not normalized:
        raise DetailLookupError("missing hash")
    if any(ch not in "0123456789abcdefABCDEF" for ch in normalized):
        raise DetailLookupError(f"invalid hash: {hash_key}")

    path = db_path or default_ccr_db_path()
    if not path.exists():
        raise DetailLookupError(f"detail store not found: {path}")

    entry_json = _lookup_entry_json(path, normalized.lower())
    if entry_json is None:
        raise DetailLookupError(f"detail not found for hash: {hash_key}")

    try:
        entry = json.loads(entry_json)
    except json.JSONDecodeError as exc:
        raise DetailLookupError(f"detail entry is corrupt for hash: {hash_key}") from exc

    original = entry.get("original_content")
    if not isinstance(original, str):
        raise DetailLookupError(f"detail entry has no original content for hash: {hash_key}")
    return original


def _lookup_entry_json(path: Path, hash_key: str) -> str | None:
    with sqlite3.connect(path) as conn:
        if len(hash_key) >= 24:
            row = conn.execute(
                "SELECT entry_json FROM ccr_entries WHERE hash = ?",
                (hash_key[:24],),
            ).fetchone()
            return str(row[0]) if row else None

        rows = conn.execute(
            "SELECT entry_json FROM ccr_entries WHERE hash LIKE ? ORDER BY created_at DESC LIMIT 2",
            (f"{hash_key}%",),
        ).fetchall()
        if len(rows) > 1:
            raise DetailLookupError(f"ambiguous hash prefix: {hash_key}")
        return str(rows[0][0]) if rows else None


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: rtk details <hash>", file=sys.stderr)
        return 2
    try:
        print(retrieve_detail(args[0]), end="")
    except DetailLookupError as exc:
        print(f"rtk details: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
