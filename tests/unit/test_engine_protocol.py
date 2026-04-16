"""Tests for taigi_asr.engines.base.ASREngine Protocol + FakeEngine."""

from __future__ import annotations

from pathlib import Path

import pytest

from taigi_asr.engines.base import ASREngine
from taigi_asr.engines.fake import FakeEngine
from taigi_asr.segments import TimestampedSegment


class TestFakeEngineSatisfiesProtocol:
    def test_fake_engine_is_structural_instance(self) -> None:
        engine: ASREngine = FakeEngine(
            script=[TimestampedSegment(0.0, 1.0, "hello")]
        )
        # Duck-typed structural check: all protocol methods exist
        assert callable(engine.load)
        assert callable(engine.transcribe)
        assert callable(engine.is_loaded)
        assert callable(engine.unload)


class TestFakeEngineBehaviour:
    def test_not_loaded_initially(self) -> None:
        e = FakeEngine(script=[])
        assert e.is_loaded() is False

    def test_load_sets_loaded_true(self) -> None:
        e = FakeEngine(script=[])
        e.load()
        assert e.is_loaded() is True

    def test_transcribe_returns_scripted_segments(self, tmp_path: Path) -> None:
        wav = tmp_path / "x.wav"
        wav.write_bytes(b"\x00" * 10)
        script = [
            TimestampedSegment(0.0, 3.0, "小時候跌倒醫沒好"),
            TimestampedSegment(3.0, 6.5, "長大傻傻在這裡喊玲瓏"),
        ]
        e = FakeEngine(script=script)
        e.load()
        out = e.transcribe(wav)
        assert out == script

    def test_transcribe_raises_when_not_loaded(self, tmp_path: Path) -> None:
        e = FakeEngine(script=[])
        wav = tmp_path / "x.wav"
        wav.write_bytes(b"\x00")
        with pytest.raises(RuntimeError):
            e.transcribe(wav)

    def test_unload_toggles_state(self) -> None:
        e = FakeEngine(script=[])
        e.load()
        e.unload()
        assert e.is_loaded() is False

    def test_tracks_call_count(self, tmp_path: Path) -> None:
        wav = tmp_path / "x.wav"
        wav.write_bytes(b"\x00")
        e = FakeEngine(script=[TimestampedSegment(0, 1, "hi")])
        e.load()
        e.transcribe(wav)
        e.transcribe(wav)
        assert e.call_count == 2
