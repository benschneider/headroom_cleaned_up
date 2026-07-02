from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sqlite3
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


details = _load_module("rtk_details", "headroom/rtk/details.py")
shim = _load_module("rtk_shim", "headroom/rtk/shim.py")
DetailLookupError = details.DetailLookupError
retrieve_detail = details.retrieve_detail
install_rtk_detail_shim = shim.install_rtk_detail_shim


def _write_store(path: Path, rows: dict[str, str]) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE ccr_entries (hash TEXT PRIMARY KEY, entry_json TEXT NOT NULL, created_at REAL NOT NULL, ttl INTEGER NOT NULL)"
        )
        for index, (hash_key, original) in enumerate(rows.items()):
            conn.execute(
                "INSERT INTO ccr_entries (hash, entry_json, created_at, ttl) VALUES (?, ?, ?, ?)",
                (
                    hash_key,
                    json.dumps({"hash": hash_key, "original_content": original}),
                    float(index),
                    1800,
                ),
            )


def test_retrieve_detail_by_full_hash(tmp_path: Path) -> None:
    db = tmp_path / "ccr_store.db"
    _write_store(db, {"a" * 24: "full output"})

    assert retrieve_detail("a" * 24, db) == "full output"


def test_retrieve_detail_by_unique_prefix(tmp_path: Path) -> None:
    db = tmp_path / "ccr_store.db"
    _write_store(db, {"abcdef" + "0" * 18: "prefixed output"})

    assert retrieve_detail("abcdef", db) == "prefixed output"


def test_retrieve_detail_rejects_ambiguous_prefix(tmp_path: Path) -> None:
    db = tmp_path / "ccr_store.db"
    _write_store(db, {"abc" + "0" * 21: "first", "abc" + "1" * 21: "second"})

    try:
        retrieve_detail("abc", db)
    except DetailLookupError as exc:
        assert "ambiguous" in str(exc)
    else:
        raise AssertionError("expected ambiguous prefix failure")


def test_rtk_shim_intercepts_details_and_delegates_other_commands(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    db = tmp_path / "ccr_store.db"
    _write_store(db, {"b" * 24: "stored detail"})
    monkeypatch.setenv("HEADROOM_CCR_SQLITE_PATH", str(db))

    rtk = tmp_path / "rtk"
    rtk.write_text("#!/bin/sh\necho real:$@\n", encoding="utf-8")
    rtk.chmod(0o755)
    install_rtk_detail_shim(rtk)

    detail = subprocess.run(
        [str(rtk), "details", "b" * 24],
        check=True,
        text=True,
        capture_output=True,
    )
    assert detail.stdout == "stored detail"

    delegated = subprocess.run(
        [str(rtk), "--version"],
        check=True,
        text=True,
        capture_output=True,
    )
    assert delegated.stdout == "real:--version\n"


def test_rtk_shim_reads_files_exactly(tmp_path: Path) -> None:
    source = tmp_path / "sample.py"
    source.write_text("def f():\n    return 1\n", encoding="utf-8")
    rtk = tmp_path / "rtk"
    rtk.write_text("#!/bin/sh\necho real:$@\n", encoding="utf-8")
    rtk.chmod(0o755)
    install_rtk_detail_shim(rtk)

    result = subprocess.run(
        [str(rtk), "read", str(source)],
        check=True,
        text=True,
        capture_output=True,
    )

    assert result.stdout == "def f():\n    return 1\n"


def test_rtk_shim_runs_git_diff_raw(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_git = bin_dir / "git"
    fake_git.write_text("#!/bin/sh\necho git:$@\n", encoding="utf-8")
    fake_git.chmod(0o755)
    import os

    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    rtk = tmp_path / "rtk"
    rtk.write_text("#!/bin/sh\necho real:$@\n", encoding="utf-8")
    rtk.chmod(0o755)
    install_rtk_detail_shim(rtk)

    diff = subprocess.run(
        [str(rtk), "git", "diff", "--", "sample.py"],
        check=True,
        text=True,
        capture_output=True,
    )
    show = subprocess.run(
        [str(rtk), "git", "show", "--stat"],
        check=True,
        text=True,
        capture_output=True,
    )
    status = subprocess.run(
        [str(rtk), "git", "status"],
        check=True,
        text=True,
        capture_output=True,
    )

    assert diff.stdout == "git:diff -- sample.py\n"
    assert show.stdout == "git:show --stat\n"
    assert status.stdout == "real:git status\n"
