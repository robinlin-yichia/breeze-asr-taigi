"""Python API usage: construct an engine directly without the router.

Useful when you want to pin a specific config (e.g. for benchmarking or
integrating into a larger pipeline).
"""

from __future__ import annotations

import sys
from pathlib import Path

from taigi_asr.audio import AudioConverter
from taigi_asr.engines.faster_whisper import FasterWhisperEngine
from taigi_asr.formatters import to_json


def main(audio_path: str) -> None:
    src = Path(audio_path)

    # Pin a specific engine config. Router would pick these automatically on
    # a 4 GB RTX 3050, but explicit construction is clearer for docs.
    engine = FasterWhisperEngine(
        device="cuda",
        compute_type="int8_float16",
        batch_size=4,
        beam_size=5,
        best_of=5,
        temperature=0.0,
    )

    wav, duration = AudioConverter.convert(src)
    try:
        engine.load()
        segments = engine.transcribe(
            wav,
            word_timestamps=False,   # flip to True for per-word timings
            beam_size=5,             # override here if you want, None=use constructor default
            vad_filter=True,
        )
    finally:
        AudioConverter.cleanup(wav)
        engine.unload()

    meta = {
        "source": str(src),
        "duration_sec": round(duration, 2),
        "compute_type": engine.compute_type,
        "batch_size": engine.batch_size,
        "engine": "faster_whisper",
        "model": FasterWhisperEngine.MODEL_ID,
    }
    out_json = src.with_suffix(".json")
    out_json.write_text(to_json(segments, meta=meta), encoding="utf-8")
    print(f"Wrote {out_json} with {len(segments)} segments")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python examples/programmatic.py <audio>")
        sys.exit(2)
    main(sys.argv[1])
