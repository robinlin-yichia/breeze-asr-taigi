"""GPU profiling and engine selection.

Auto-routes to the most capable engine that fits the detected VRAM budget,
with a user-override escape hatch. See docs/architecture.md for the full
decision table.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from taigi_asr.errors import InsufficientVRAMError


def _bitsandbytes_available() -> bool:
    """Probe whether bitsandbytes can actually load on this machine.

    bitsandbytes ships Linux-only CUDA binaries upstream, but community
    Windows wheels exist and users sometimes install them. We just try to
    import and actually touch the library so any missing CUDA DLL surfaces
    as an ImportError/OSError — a failed import is the real signal, not the
    platform string.
    """
    try:
        import bitsandbytes  # noqa: F401
    except Exception:
        return False
    return True


class EngineKind(str, Enum):
    FASTER_WHISPER = "faster_whisper"
    HUGGINGFACE = "huggingface"


@dataclass(frozen=True, slots=True)
class GPUInfo:
    name: str
    vram_gb: float
    cuda_available: bool
    bf16_supported: bool


@dataclass(frozen=True, slots=True)
class EngineSpec:
    """Parameters the caller hands to a concrete engine.

    Attributes are intentionally low-level (compute_type string matches
    faster-whisper's vocabulary) so the router stays decoupled from engine
    impl details.
    """

    kind: EngineKind
    device: str  # "cuda" | "cpu"
    compute_type: str  # faster-whisper vocab: float16 | int8_float16 | int8
    batch_size: int


class GPUProfiler:
    """Thin wrapper around torch.cuda — mockable in tests."""

    @staticmethod
    def detect() -> GPUInfo:
        try:
            import torch  # imported lazily so unit tests can patch before import
        except ImportError:
            return GPUInfo(name="CPU", vram_gb=0.0, cuda_available=False, bf16_supported=False)

        if not torch.cuda.is_available():
            return GPUInfo(name="CPU", vram_gb=0.0, cuda_available=False, bf16_supported=False)

        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024**3)
        try:
            bf16 = bool(torch.cuda.is_bf16_supported())
        except Exception:
            bf16 = False
        return GPUInfo(
            name=name,
            vram_gb=round(vram_gb, 2),
            cuda_available=True,
            bf16_supported=bf16,
        )


class EngineRouter:
    """Deterministic VRAM -> engine mapping."""

    @staticmethod
    def select(
        info: GPUInfo,
        prefer: EngineKind | None = None,
        strict: bool = False,
    ) -> EngineSpec:
        """Pick the best engine for the hardware.

        Args:
            info: GPU profile.
            prefer: Force a specific engine. Ignored if it exceeds VRAM and
                ``strict`` is False (auto-downgrades to faster-whisper).
            strict: If True and the preferred engine cannot fit, raise
                :class:`InsufficientVRAMError` instead of downgrading.
        """
        # CPU-only path
        if not info.cuda_available:
            return EngineSpec(
                kind=EngineKind.FASTER_WHISPER,
                device="cpu",
                compute_type="int8",
                batch_size=1,
            )

        vram = info.vram_gb

        # Resolve override
        if prefer is EngineKind.HUGGINGFACE:
            if vram < 3.5:
                if strict:
                    raise InsufficientVRAMError(
                        f"HuggingFace engine needs >= 3.5 GB VRAM even in aggressive "
                        f"low-memory mode; detected {vram:.1f} GB."
                    )
                return EngineRouter._fw_spec(vram)
            return EngineRouter._hf_spec(vram)

        if prefer is EngineKind.FASTER_WHISPER:
            return EngineRouter._fw_spec(vram)

        # Auto — always Faster-Whisper, regardless of how much VRAM is free.
        #
        # Measured on Breeze-ASR-26, RTX 3080 Ti 12 GB, 120 s of meeting audio
        # (characters counted with whitespace stripped, so segmentation does
        # not skew the comparison):
        #
        #     engine                chars  segments   wall
        #     FW  int8_float16        529        21    3.3 s
        #     HF  float16             520        60   57.0 s
        #
        # HF is 17x slower for slightly less text. Two compounding reasons:
        # ``torch.compile``, which _hf_spec's throughput assumes, silently
        # no-ops on Windows (inductor needs triton, which has no Windows
        # wheel), and transformers' word-timestamp alignment runs DTW over
        # per-layer cross-attention retained for the whole file. That last
        # part also makes memory grow with duration: a 2.5 h recording died
        # with CUDA OOM after 138 minutes at 11998/12288 MiB, while the
        # CTranslate2 path finished the same file in 230 s.
        #
        # Bigger cards do not change the ranking — they only postpone the OOM
        # — so there is no VRAM threshold above which HF becomes the better
        # automatic choice. It stays reachable via ``prefer``.
        return EngineRouter._fw_spec(vram)

    @staticmethod
    def _fw_spec(vram: float) -> EngineSpec:
        # Windows reports RTX 3050 Laptop as ~3.84 GB. Use 3.5 GB as the 4 GB
        # class threshold so Optimus/hybrid-GPU laptops route to the same
        # batch=4 setting that 4 GB dedicated cards use.
        if vram >= 14.0:
            batch, ct = 8, "float16"
        elif vram >= 3.5:
            batch, ct = 4, "int8_float16"
        else:
            batch, ct = 2, "int8_float16"
        return EngineSpec(
            kind=EngineKind.FASTER_WHISPER,
            device="cuda",
            compute_type=ct,
            batch_size=batch,
        )

    @staticmethod
    def _hf_spec(vram: float) -> EngineSpec:
        # VRAM peak for Breeze-ASR-26 (Whisper-large-v2, 2B params) in fp16:
        #   model weights      ~3.1 GB
        #   encoder activations 0.3-0.5 GB
        #   decoder KV cache    0.3-0.6 GB
        # On Optimus laptops the dGPU is dedicated to CUDA (no desktop compositor
        # stealing VRAM), so 4 GB cards can just barely fit fp16 with batch=1.
        if vram >= 22.0:
            batch, ct = 16, "float16"
        elif vram >= 14.0:
            batch, ct = 8, "float16"
        elif vram >= 10.0:
            batch, ct = 4, "float16"
        elif vram >= 6.0:
            # bitsandbytes 8-bit is Linux-only; without it the HF path OOMs on
            # 6-10 GB cards. Use fp16 batch=1 instead — slower, but at least
            # runs on Windows.
            if _bitsandbytes_available():
                batch, ct = 2, "int8"
            else:
                batch, ct = 1, "float16"
        else:
            batch, ct = 1, "float16"  # 4 GB dGPU: aggressive low-mem
        return EngineSpec(
            kind=EngineKind.HUGGINGFACE,
            device="cuda",
            compute_type=ct,
            batch_size=batch,
        )
