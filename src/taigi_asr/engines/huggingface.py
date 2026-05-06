"""HuggingFace Transformers pipeline engine.

Runs MediaTek-Research/Breeze-ASR-26 directly via the
``AutomaticSpeechRecognitionPipeline``. Best on 8 GB+ GPUs; falls back to
bitsandbytes 8-bit quantization for 6–8 GB. Below 6 GB the router should have
already routed to :class:`FasterWhisperEngine` instead.
"""

from __future__ import annotations

import gc
import logging
import os
import threading
from pathlib import Path

# Low-VRAM allocator hint — set as early as possible, before torch picks up the
# CUDA context. No-op if torch already initialized, but harmless.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from taigi_asr.config import (
    DEFAULT_CHUNK_LENGTH_S,
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_NO_REPEAT_NGRAM_SIZE,
    DEFAULT_STRIDE_LENGTH_S,
    HUGGINGFACE_MODEL_ID,
)
from taigi_asr.errors import InsufficientVRAMError, ModelLoadError, TranscriptionError
from taigi_asr.segments import TimestampedSegment

log = logging.getLogger(__name__)


class HuggingFaceEngine:
    """Whisper pipeline with torch.compile + SDPA + optional 8-bit quant."""

    MODEL_ID: str = HUGGINGFACE_MODEL_ID

    def __init__(
        self,
        device: str = "cuda",
        compute_type: str = "float16",  # "float16" or "int8" (bitsandbytes)
        batch_size: int = 4,
        chunk_length_s: int = DEFAULT_CHUNK_LENGTH_S,
    ) -> None:
        self.device = device
        self.compute_type = compute_type
        self.batch_size = batch_size
        self.chunk_length_s = chunk_length_s
        self.pipe = None
        self._loaded = False
        self._lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            try:
                import torch
                from transformers import (
                    AutoModelForSpeechSeq2Seq,
                    AutoProcessor,
                    pipeline,
                )
            except ImportError as exc:
                raise ModelLoadError(
                    'transformers / torch not installed. Run: pip install -e ".[hf]"'
                ) from exc

            try:
                processor = AutoProcessor.from_pretrained(self.MODEL_ID)

                model_kwargs: dict = {
                    "low_cpu_mem_usage": True,
                    "attn_implementation": "sdpa" if self.device != "cpu" else "eager",
                }

                dtype = torch.float16
                if self.compute_type == "int8":
                    # bitsandbytes 8-bit quantization — Linux only.
                    try:
                        from transformers import BitsAndBytesConfig

                        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                    except ImportError as exc:
                        raise InsufficientVRAMError(
                            "HuggingFace int8 path requires bitsandbytes, which failed "
                            "to import. Fall back to the FasterWhisperEngine."
                        ) from exc
                else:
                    model_kwargs["torch_dtype"] = dtype

                model = AutoModelForSpeechSeq2Seq.from_pretrained(self.MODEL_ID, **model_kwargs)
                if self.compute_type != "int8":
                    model.to(self.device)

                # torch.compile is best-effort — unavailable on Windows / PyTorch builds.
                try:
                    model = torch.compile(model, mode="reduce-overhead")
                except Exception:
                    log.info("torch.compile unavailable; using eager mode.")

                self.pipe = pipeline(
                    "automatic-speech-recognition",
                    model=model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    torch_dtype=dtype if self.compute_type != "int8" else None,
                    device=self.device if self.compute_type != "int8" else None,
                    chunk_length_s=self.chunk_length_s,
                    stride_length_s=DEFAULT_STRIDE_LENGTH_S,
                    return_timestamps=True,
                )

                # CUDA warmup to amortize JIT kernel compilation.
                if self.device != "cpu":
                    try:
                        import numpy as np

                        dummy = np.zeros(16000, dtype=np.float32)
                        self.pipe(
                            dummy,
                            generate_kwargs={
                                "language": DEFAULT_LANGUAGE,
                                "task": "transcribe",
                                "max_new_tokens": 10,
                            },
                        )
                    except Exception:
                        log.debug("CUDA warmup skipped.")

                self._loaded = True
            except InsufficientVRAMError:
                raise
            except Exception as exc:
                # Critical: drop any partially-loaded model reference so VRAM
                # is reclaimed before we re-raise. Otherwise the pipeline /
                # model weights linger on the GPU even though load() failed.
                self.pipe = None
                try:
                    import torch

                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
                raise ModelLoadError(f"Failed to load HuggingFace model: {exc}") from exc

    def is_loaded(self) -> bool:
        return self._loaded

    def transcribe(
        self,
        wav_path: str | Path,
        *,
        word_timestamps: bool = False,
    ) -> list[TimestampedSegment]:
        if not self._loaded:
            self.load()
        assert self.pipe is not None

        try:
            result = self.pipe(
                str(wav_path),
                batch_size=self.batch_size,
                generate_kwargs={
                    "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
                    "language": DEFAULT_LANGUAGE,
                    "task": "transcribe",
                    "no_repeat_ngram_size": DEFAULT_NO_REPEAT_NGRAM_SIZE,
                },
                return_timestamps="word" if word_timestamps else True,
            )
        except Exception as exc:
            raise TranscriptionError(f"HuggingFace pipeline failed: {exc}") from exc

        out: list[TimestampedSegment] = []
        chunks = result.get("chunks")
        if isinstance(chunks, list):
            for chunk in chunks:
                text = (chunk.get("text") or "").strip()
                if not text:
                    continue
                ts = chunk.get("timestamp")
                if isinstance(ts, tuple) and len(ts) >= 2:
                    start = float(ts[0]) if ts[0] is not None else 0.0
                    end = float(ts[1]) if ts[1] is not None else start + 1.0
                else:
                    start, end = 0.0, 0.0
                out.append(TimestampedSegment(start, end, text))
        else:
            text = (result.get("text") or "").strip()
            if text:
                out.append(TimestampedSegment(0.0, 0.0, text))
        return out

    def unload(self) -> None:
        """Release VRAM held by the HuggingFace pipeline.

        Dropping ``self.pipe`` alone is not enough — the pipeline internally
        retains a strong reference to the Whisper model. We explicitly delete
        both the model and the pipeline, run ``gc.collect()`` to force the
        refcounted weights out of scope, then empty the CUDA allocator cache.
        """
        with self._lock:
            if self.pipe is not None:
                model = getattr(self.pipe, "model", None)
                self.pipe = None
                if model is not None:
                    del model
            self._loaded = False
            try:
                import torch

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
