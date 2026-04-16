"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def tone_wav(tmp_path: Path) -> Path:
    """Generate a 2-second, 44.1kHz, stereo sine wave — forces conversion path."""
    sr = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    left = 0.1 * np.sin(2 * np.pi * 440 * t)
    right = 0.1 * np.sin(2 * np.pi * 880 * t)
    stereo = np.stack([left, right], axis=1).astype(np.float32)
    path = tmp_path / "tone.wav"
    sf.write(path, stereo, sr, subtype="FLOAT")
    return path


@pytest.fixture
def project_root() -> Path:
    """Repo root path, useful for locating data/test.m4a in integration tests."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def sample_audio(project_root: Path) -> Path:
    """The canonical Hokkien sample audio (data/test.m4a)."""
    path = project_root / "data" / "test.m4a"
    if not path.exists():
        pytest.skip(f"Sample audio missing: {path}")
    return path
