"""Unit tests for ``taigi_asr.cli._resolve_inputs``.

Covers the new multi-file batch surface — pure-function tests so they don't
need a GPU, ffmpeg, or the model. The CLI's outer flow (argparse, engine
load, format dispatch) is exercised separately via the smoke + integration
suites.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from taigi_asr.cli import _resolve_inputs
from taigi_asr.config import SUPPORTED_AUDIO_EXTS


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_positional_only_preserves_order(tmp_path: Path) -> None:
    a = _touch(tmp_path / "a.mp3")
    b = _touch(tmp_path / "b.m4a")
    c = _touch(tmp_path / "c.wav")
    assert _resolve_inputs([a, b, c], None) == [a, b, c]


def test_positional_dedups_repeats(tmp_path: Path) -> None:
    a = _touch(tmp_path / "song.mp3")
    # Same physical file given twice — the second mention drops out.
    assert _resolve_inputs([a, a], None) == [a]


def test_positional_dedups_relative_vs_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Same file referenced via relative + absolute paths must collapse to one.
    monkeypatch.chdir(tmp_path)
    abs_path = _touch(tmp_path / "x.mp3")
    rel_path = Path("x.mp3")
    out = _resolve_inputs([rel_path, abs_path], None)
    assert len(out) == 1


def test_input_dir_picks_supported_extensions(tmp_path: Path) -> None:
    music = tmp_path / "music"
    _touch(music / "track.mp3")
    _touch(music / "voice.m4a")
    _touch(music / "loop.wav")
    _touch(music / "notes.txt")  # unsupported — must be ignored
    _touch(music / "cover.png")  # unsupported — must be ignored

    out = _resolve_inputs([], music)
    names = sorted(p.name for p in out)
    assert names == ["loop.wav", "track.mp3", "voice.m4a"]


def test_input_dir_extensions_are_case_insensitive(tmp_path: Path) -> None:
    music = tmp_path / "music"
    _touch(music / "LOUD.MP3")
    _touch(music / "Quiet.M4A")
    out = _resolve_inputs([], music)
    assert len(out) == 2
    assert {p.name for p in out} == {"LOUD.MP3", "Quiet.M4A"}


def test_positional_plus_input_dir_merges_without_duplicates(tmp_path: Path) -> None:
    music = tmp_path / "music"
    a = _touch(music / "a.mp3")
    _touch(music / "b.mp3")  # picked up by --input-dir
    extra = _touch(tmp_path / "extra.flac")

    # `a` is named explicitly AND lives inside the dir glob — should appear
    # exactly once. Positional listings come before dir-glob entries.
    out = _resolve_inputs([a, extra], music)
    names = [p.name for p in out]
    assert names == ["a.mp3", "extra.flac", "b.mp3"]


def test_input_dir_skips_subdirectories(tmp_path: Path) -> None:
    music = tmp_path / "music"
    _touch(music / "top.mp3")
    _touch(music / "nested" / "deep.mp3")  # in a subdir
    out = _resolve_inputs([], music)
    assert [p.name for p in out] == ["top.mp3"]


def test_missing_input_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _resolve_inputs([], tmp_path / "does-not-exist")


def test_input_dir_pointing_at_file_raises(tmp_path: Path) -> None:
    f = _touch(tmp_path / "x.mp3")
    with pytest.raises(NotADirectoryError):
        _resolve_inputs([], f)


def test_empty_dir_returns_empty_list(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _resolve_inputs([], empty) == []


def test_audio_ext_whitelist_is_lowercase() -> None:
    # If anyone adds an uppercase entry the case-insensitive matching breaks
    # silently — guard against that with a tiny invariant test.
    for ext in SUPPORTED_AUDIO_EXTS:
        assert ext == ext.lower(), f"SUPPORTED_AUDIO_EXTS must be lowercase: {ext}"


def test_audio_ext_whitelist_starts_with_dot() -> None:
    # ``Path.suffix`` always returns the extension *with* the leading dot.
    # If someone forgets the dot the membership check will silently never
    # match — catch that here.
    for ext in SUPPORTED_AUDIO_EXTS:
        assert ext.startswith("."), f"SUPPORTED_AUDIO_EXTS entries need leading dot: {ext}"
