"""CLI smoke tests — just verify argparse wiring without running inference."""

from __future__ import annotations

import subprocess
import sys

import pytest


def test_cli_help_exits_zero() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "taigi_asr.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0
    assert "taigi-asr" in result.stdout
    assert "--engine" in result.stdout
    assert "--format" in result.stdout


def test_cli_version() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "taigi_asr.cli", "--version"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0
    assert "taigi-asr" in result.stdout


def test_cli_missing_audio_exits_nonzero(tmp_path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "taigi_asr.cli", str(tmp_path / "nope.m4a")],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode != 0
    assert "not found" in result.stderr.lower()


@pytest.mark.parametrize("engine_flag", ["auto", "fw", "faster-whisper", "hf", "huggingface"])
def test_cli_accepts_known_engines(engine_flag: str, tmp_path) -> None:
    # Only check that parser accepts it — shouldn't execute inference
    # since audio file is missing; exit code 2 or later is fine.
    result = subprocess.run(
        [sys.executable, "-m", "taigi_asr.cli", str(tmp_path / "missing.wav"), "--engine", engine_flag],
        capture_output=True,
        text=True,
        timeout=15,
    )
    # Must not exit due to argparse failure (which would be code 2 with
    # 'invalid choice' in stderr)
    assert "invalid choice" not in result.stderr
    assert "Unknown engine" not in result.stderr
