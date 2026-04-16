# Architecture

## Module dependency graph

```
segments  <-  (pure leaf, no deps)
errors    <-  (pure leaf)
config    <-  stdlib only
formatters -> segments
audio     <-  stdlib + soundfile/pydub lazily
router    -> errors  (torch imported lazily inside GPUProfiler.detect)
engines/
  base     -> segments
  fake     -> segments
  faster_whisper -> config, errors, segments  (faster_whisper lazy import)
  huggingface    -> config, errors, segments  (torch/transformers lazy import)
  __init__ -> base, fake, errors, router
cli       -> audio, engines, errors, formatters, router, segments
ui/
  gradio_app -> audio, engines, errors, formatters, router, segments
  launcher   -> config, ui.gradio_app
```

No cycles. `segments`/`errors`/`config` are pure Python leaves. `audio` / `formatters` do not import torch. `router` imports torch only inside a method (mockable in tests). Concrete engines pull torch/faster-whisper lazily so CI runners without CUDA can still import the package.

## Request flow

```
user audio file
     |
     v
AudioConverter.convert        (ffmpeg -> 16 kHz mono wav, +pydub fallback)
     |
     v
GPUProfiler.detect            (nvidia VRAM probe, torch.cuda mock seam)
     |
     v
EngineRouter.select           (VRAM -> EngineSpec)
     |
     v
build_engine(spec)            (factory -> FasterWhisperEngine | HuggingFaceEngine)
     |
     v
engine.load()                 (download + load + warmup; lock-guarded, idempotent)
     |
     v
engine.transcribe(wav_path)   (beam search + VAD + batched chunks)
     |
     v
list[TimestampedSegment]
     |
     v
to_srt / to_txt / to_vtt / to_json
     |
     v
disk / browser download
```

## Engine Protocol

```python
class ASREngine(Protocol):
    def load(self) -> None: ...
    def is_loaded(self) -> bool: ...
    def transcribe(self, wav_path, *, word_timestamps=False) -> list[TimestampedSegment]: ...
    def unload(self) -> None: ...
```

All three engines (`FasterWhisperEngine`, `HuggingFaceEngine`, `FakeEngine`) satisfy the Protocol. Callers hold `ASREngine` references, never concrete types. `build_engine(spec)` is the only place where the spec enum maps to a concrete class.

## VRAM routing

See `src/taigi_asr/router.py::EngineRouter._fw_spec` / `_hf_spec`. Key design choices:

- **RTX 3050 Laptop 4GB = main target**. Windows reports ~3.84 GB; the threshold is 3.5 GB so the 3050 class lands in the safe int8_float16 + batch=4 bucket.
- **int8_float16 is preferred over float16 on Ampere consumer cards**. Tensor Cores are 8-bit optimal; empirically `int8_float16` is both faster AND more VRAM-efficient than `float16` on the 3050.
- **HF path requires bitsandbytes for < 8 GB cards**. bitsandbytes is Linux-only, so on Windows the router auto-downgrades to Faster-Whisper for 4-6 GB cards.
- **CPU fallback uses `compute_type="int8"`** (pure int8, no fp16 mixing).

## VRAM safety

Peak VRAM for Breeze-ASR-26 (Whisper-large-v2, ~1.54B params):

| compute_type | Model | Activations (chunk=30s) | KV cache | Peak |
|---|---|---|---|---|
| float16 | ~3.1 GB | ~0.3-0.5 GB | ~0.3-0.6 GB | ~3.9-4.5 GB |
| int8_float16 | ~1.6 GB | ~0.3-0.4 GB | ~0.3-0.5 GB | ~2.5-3.0 GB |
| int8 (bitsandbytes) | ~1.6 GB | ~0.3-0.4 GB | ~0.3-0.5 GB | ~2.5-3.0 GB |

Safety mechanisms:
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set in both engine modules at import time, before torch grabs the CUDA context.
- UI `_engine_cache` is an LRU-of-1: switching engine kinds triggers `unload()` on the previous instance before loading the new one. Prevents two Whisper-large models coexisting in VRAM.
- `unload()` explicitly `del`s the pipeline and model, then `gc.collect()` + `torch.cuda.empty_cache()`. Dropping `self.pipe = None` alone does not free VRAM because the HF pipeline retains a strong reference to the model internally.
- Engine construction is lazy — `build_engine(spec)` does not touch the GPU until `load()` is called. Safe to cache multiple spec variants without VRAM cost.

## Thread safety

- `engines/base.py` Protocol does not mandate thread safety.
- `FasterWhisperEngine.load()` and `unload()` are protected by `self._lock`. `transcribe()` reads `self._loaded` under the lock then auto-loads outside the lock (documented TOCTOU: double-load prevented by the inner `_lock` in `load()`).
- UI `_get_or_build_engine` is protected by `_engine_lock` so concurrent Gradio requests cannot race on cache eviction.
- `AudioConverter` is stateless; `convert()` is safe to call concurrently, each call gets its own temp file.

## Test pyramid

```
tests/
  unit/          <-  no GPU, no model download (<3s total)
    test_segments.py          20 tests
    test_formatters.py        11 tests
    test_audio.py              6 tests (needs ffmpeg; skipped otherwise)
    test_router.py            14 tests (torch.cuda mocked)
    test_engine_protocol.py    7 tests (uses FakeEngine)
  smoke/         <-  CLI + UI importability (~10s)
    test_gradio_starts.py    2 tests
    test_cli_help.py         8 tests
  integration/   <-  real model + real audio; marked pytest.mark.slow
    test_faster_whisper_engine.py  2 tests
    test_end_to_end.py             2 tests (1 with FakeEngine, 1 with real)
```

CI runs `pytest tests/unit tests/smoke` (no GPU). Local/contributor runs `pytest -m slow` for integration.

## Error taxonomy

```
Exception
 └── TaigiASRError                      (all raises go through this base)
      ├── InsufficientVRAMError          (router / engine refuses to load)
      ├── ModelLoadError                 (HF download / CT2 init failed)
      └── TranscriptionError             (inference-time exception)
```

UI catches `InsufficientVRAMError` for auto-downgrade to Faster-Whisper. CLI maps each subtype to a distinct exit code (3 for VRAM, 4 for everything else).
