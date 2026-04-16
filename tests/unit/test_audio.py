"""Tests for taigi_asr.audio.AudioConverter."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import soundfile as sf

from taigi_asr.audio import AudioConverter


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


class TestAudioConverter:
    def test_target_sample_rate_is_16khz(self) -> None:
        assert AudioConverter.TARGET_SR == 16000

    @pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not installed")
    def test_convert_resamples_to_16k_mono(self, tone_wav: Path, tmp_path: Path) -> None:
        wav_path, duration = AudioConverter.convert(tone_wav, out_dir=tmp_path)
        info = sf.info(wav_path)
        assert info.samplerate == 16000
        assert info.channels == 1
        assert duration == pytest.approx(2.0, abs=0.05)
        assert Path(wav_path).exists()
        assert Path(wav_path).suffix == ".wav"

    @pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not installed")
    def test_convert_raises_on_missing_source(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            AudioConverter.convert(tmp_path / "does-not-exist.mp3")

    @pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not installed")
    def test_convert_output_is_16bit_pcm(self, tone_wav: Path, tmp_path: Path) -> None:
        wav_path, _ = AudioConverter.convert(tone_wav, out_dir=tmp_path)
        info = sf.info(wav_path)
        # pcm_s16le -> soundfile reports 'PCM_16'
        assert "PCM_16" in info.subtype or info.subtype_info == "Signed 16 bit PCM"

    def test_cleanup_removes_temp_file(self, tmp_path: Path) -> None:
        # If a file with a wav suffix is passed to cleanup, it should be removed.
        fake_wav = tmp_path / "will_be_gone.wav"
        fake_wav.write_bytes(b"RIFF")
        AudioConverter.cleanup(fake_wav)
        assert not fake_wav.exists()

    def test_cleanup_handles_missing_file_quietly(self, tmp_path: Path) -> None:
        # Must not raise on a non-existent path.
        AudioConverter.cleanup(tmp_path / "ghost.wav")
