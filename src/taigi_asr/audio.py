"""Audio ingest and normalization.

Converts any ffmpeg-readable audio/video into 16 kHz mono PCM s16le WAV — the format
Whisper / faster-whisper expect. Falls back to pydub (pure-Python decode via libavcodec
bindings) when the ffmpeg CLI is unavailable.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


class AudioConverter:
    """Static utility: convert arbitrary audio to 16 kHz mono 16-bit WAV."""

    TARGET_SR: int = 16000

    @staticmethod
    def convert(src_path: str | Path, out_dir: str | Path | None = None) -> tuple[Path, float]:
        """Normalize to 16 kHz mono 16-bit PCM WAV.

        Returns (wav_path, duration_seconds). Caller is responsible for calling
        :meth:`cleanup` on the returned path once finished.
        """
        src = Path(src_path)
        if not src.exists():
            raise FileNotFoundError(f"Source audio not found: {src}")

        out_dir_path = Path(out_dir) if out_dir is not None else Path(tempfile.gettempdir())
        out_dir_path.mkdir(parents=True, exist_ok=True)
        fd, raw = tempfile.mkstemp(suffix=".wav", dir=out_dir_path)
        os.close(fd)
        wav_path = Path(raw)

        if shutil.which("ffmpeg"):
            cmd = [
                "ffmpeg",
                "-y",
                "-threads",
                str(os.cpu_count() or 4),
                "-i",
                str(src),
                "-ar",
                str(AudioConverter.TARGET_SR),
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                "-loglevel",
                "error",
                str(wav_path),
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError:
                AudioConverter._convert_via_pydub(src, wav_path)
        else:
            AudioConverter._convert_via_pydub(src, wav_path)

        try:
            import soundfile as sf

            duration = float(sf.info(wav_path).duration)
        except Exception:
            duration = 0.0

        return wav_path, duration

    @staticmethod
    def _convert_via_pydub(src: Path, wav_out: Path) -> None:
        from pydub import AudioSegment

        audio = AudioSegment.from_file(src)
        audio = (
            audio.set_frame_rate(AudioConverter.TARGET_SR).set_channels(1).set_sample_width(2)
        )
        audio.export(wav_out, format="wav")

    @staticmethod
    def cleanup(wav_path: str | Path) -> None:
        """Idempotently delete a temporary wav produced by :meth:`convert`."""
        p = Path(wav_path)
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass
