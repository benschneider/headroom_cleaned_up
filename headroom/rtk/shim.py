"""Install a Headroom-owned shim in front of the upstream RTK binary."""

from __future__ import annotations

from pathlib import Path
import os
import stat


SHIM_MARKER = "# headroom-rtk-detail-shim"
DETAIL_COMMANDS = {"detail", "details", "show", "expand", "retrieve"}
RAW_COMMANDS = {"cat", "diff", "nl", "sed"}
RAW_GIT_COMMANDS = {"diff", "show"}


def install_rtk_detail_shim(rtk_path: Path) -> Path:
    """Wrap an RTK binary so `rtk details <hash>` can recover CCR originals."""

    rtk_path = rtk_path.expanduser()
    real_path = rtk_path.with_name(_real_binary_name(rtk_path))
    rtk_path.parent.mkdir(parents=True, exist_ok=True)

    if rtk_path.exists() and not _is_headroom_shim(rtk_path):
        if real_path.exists():
            real_path.unlink()
        rtk_path.rename(real_path)

    script = _shim_script(real_path.name)
    rtk_path.write_text(script, encoding="utf-8")
    rtk_path.chmod(rtk_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return real_path


def _real_binary_name(rtk_path: Path) -> str:
    return "rtk-real.exe" if rtk_path.name.endswith(".exe") else "rtk-real"


def _is_headroom_shim(path: Path) -> bool:
    try:
        return SHIM_MARKER in path.read_text(encoding="utf-8", errors="ignore")[:256]
    except OSError:
        return False


def _shim_script(real_name: str) -> str:
    commands = ", ".join(repr(command) for command in sorted(DETAIL_COMMANDS))
    raw_commands = ", ".join(repr(command) for command in sorted(RAW_COMMANDS))
    raw_git_commands = ", ".join(repr(command) for command in sorted(RAW_GIT_COMMANDS))
    return f"""#!/usr/bin/env python3
{SHIM_MARKER}
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys

DETAIL_COMMANDS = {{{commands}}}
RAW_COMMANDS = {{{raw_commands}}}
RAW_GIT_COMMANDS = {{{raw_git_commands}}}


def default_db() -> Path:
    configured = os.environ.get("HEADROOM_CCR_SQLITE_PATH")
    return Path(configured).expanduser() if configured else Path.home() / ".headroom" / "ccr_store.db"


def lookup_detail(hash_key: str) -> str:
    normalized = hash_key.strip().lower()
    if not normalized:
        raise RuntimeError("missing hash")
    if any(ch not in "0123456789abcdef" for ch in normalized):
        raise RuntimeError(f"invalid hash: {{hash_key}}")
    db = default_db()
    if not db.exists():
        raise RuntimeError(f"detail store not found: {{db}}")
    with sqlite3.connect(db) as conn:
        if len(normalized) >= 24:
            row = conn.execute("SELECT entry_json FROM ccr_entries WHERE hash = ?", (normalized[:24],)).fetchone()
        else:
            rows = conn.execute(
                "SELECT entry_json FROM ccr_entries WHERE hash LIKE ? ORDER BY created_at DESC LIMIT 2",
                (f"{{normalized}}%",),
            ).fetchall()
            if len(rows) > 1:
                raise RuntimeError(f"ambiguous hash prefix: {{hash_key}}")
            row = rows[0] if rows else None
    if not row:
        raise RuntimeError(f"detail not found for hash: {{hash_key}}")
    entry = json.loads(row[0])
    original = entry.get("original_content")
    if not isinstance(original, str):
        raise RuntimeError(f"detail entry has no original content for hash: {{hash_key}}")
    return original


def print_files(paths: list[str]) -> int:
    if not paths:
        print("usage: rtk read <file> [file...]", file=sys.stderr)
        return 2
    for index, path in enumerate(paths):
        if path == "-":
            content = sys.stdin.read()
        else:
            content = Path(path).read_text(encoding="utf-8", errors="replace")
        if index:
            print()
        print(content, end="" if content.endswith("\\n") else "\\n")
    return 0


def exec_raw(command: str, args: list[str]) -> int:
    binary = shutil.which(command)
    if binary is None:
        print(f"rtk: raw command not found: {{command}}", file=sys.stderr)
        return 127
    os.execv(binary, [binary, *args])
    return 127


def main() -> int:
    args = sys.argv[1:]
    if args and args[0] in DETAIL_COMMANDS:
        if len(args) != 2:
            print("usage: rtk details <hash>", file=sys.stderr)
            return 2
        try:
            print(lookup_detail(args[1]), end="")
        except Exception as exc:
            print(f"rtk details: {{exc}}", file=sys.stderr)
            return 1
        return 0

    if args and args[0] == "--raw":
        if len(args) < 2:
            print("usage: rtk --raw <command> [args...]", file=sys.stderr)
            return 2
        return exec_raw(args[1], args[2:])

    if args and args[0] == "read":
        return print_files(args[1:])

    if args and args[0] in RAW_COMMANDS:
        return exec_raw(args[0], args[1:])

    if len(args) >= 2 and args[0] == "git" and args[1] in RAW_GIT_COMMANDS:
        return exec_raw("git", args[1:])

    real = os.path.join(os.path.dirname(os.path.realpath(__file__)), {real_name!r})
    if not os.path.exists(real):
        print(f"rtk: real binary not found: {{real}}", file=sys.stderr)
        return 127
    os.execv(real, [real, *args])
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
"""
