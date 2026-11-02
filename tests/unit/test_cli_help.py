"""LLD-47: CLI help and command documentation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from src.ingestion.pipeline import main

REPO_ROOT = Path(__file__).resolve().parents[2]


def _help(argv: list[str], capsys) -> tuple[int, str]:
    with pytest.raises(SystemExit) as exc:
        main(argv)
    return int(exc.value.code or 0), capsys.readouterr().out


def test_top_level_help_lists_commands_storage_and_exit_codes(capsys) -> None:
    code, out = _help(["--help"], capsys)
    assert code == 0
    assert "run" in out
    assert "bulk" in out
    assert "discovery CSV" in out
    assert "data/corpus/raw" in out
    assert "ROOT/<sha256>.pdf" in out
    assert "130" in out
    assert "unresolved/accounting gap" in out


def test_run_and_bulk_help_document_their_candidate_formats(capsys) -> None:
    code, out = _help(["run", "--help"], capsys)
    assert code == 0
    assert "run: JSON list of candidate objects" in out
    assert "bulk: discovery CSV with a required 'url' column" in out

    code, out = _help(["bulk", "--help"], capsys)
    assert code == 0
    assert "bulk" in out
    assert "discovery CSV with a required 'url' column" in out
    assert "resume after Ctrl-C" in out


def test_run_rejects_candidates_and_url_together(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["run", "--candidates", "x.json", "--url", "https://example.test/x.pdf"])
    assert exc.value.code == 2
    assert "mutually exclusive" in capsys.readouterr().err


def test_bulk_rejects_url_and_requires_candidates(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["bulk", "--url", "https://example.test/x.pdf"])
    assert exc.value.code == 2
    assert "bulk does not accept --url" in capsys.readouterr().err

    with pytest.raises(SystemExit) as exc:
        main(["bulk"])
    assert exc.value.code == 2
    assert "bulk requires --candidates PATH" in capsys.readouterr().err


def test_module_bulk_help_matches_parser() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "src.ingestion.pipeline", "bulk", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0
    assert "discovery CSV" in proc.stdout
    assert "resume after Ctrl-C" in proc.stdout
    assert proc.stderr == ""
