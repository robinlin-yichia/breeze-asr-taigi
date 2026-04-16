"""Tests for taigi_asr.router: VRAM-based engine selection."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from taigi_asr.router import EngineKind, EngineRouter, EngineSpec, GPUInfo, GPUProfiler


class TestGPUProfiler:
    def test_cpu_only_when_no_cuda(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            info = GPUProfiler.detect()
        assert info.cuda_available is False
        assert info.vram_gb == 0.0
        assert info.name == "CPU"

    def test_reports_vram_and_name(self) -> None:
        fake_props = type("P", (), {"total_memory": 4 * 1024**3, "major": 8, "minor": 6})()
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_name", return_value="NVIDIA GeForce RTX 3050 Laptop GPU"),
            patch("torch.cuda.get_device_properties", return_value=fake_props),
            patch("torch.cuda.is_bf16_supported", return_value=False),
        ):
            info = GPUProfiler.detect()
        assert info.cuda_available is True
        assert info.vram_gb == pytest.approx(4.0, abs=0.01)
        assert "RTX 3050" in info.name
        assert info.bf16_supported is False


class TestEngineRouter:
    @pytest.mark.parametrize(
        ("vram_gb", "cuda", "expected_kind", "expected_compute"),
        [
            (0.0, False, EngineKind.FASTER_WHISPER, "int8"),  # CPU-only
            (2.0, True, EngineKind.FASTER_WHISPER, "int8_float16"),  # tiny GPU
            (3.84, True, EngineKind.FASTER_WHISPER, "int8_float16"),  # RTX 3050 Laptop (Windows)
            (4.0, True, EngineKind.FASTER_WHISPER, "int8_float16"),  # 4 GB exact
            (6.0, True, EngineKind.HUGGINGFACE, "int8"),  # 6GB -> HF int8
            (8.0, True, EngineKind.HUGGINGFACE, "int8"),  # 8GB -> HF int8 (fp16 too tight)
            (12.0, True, EngineKind.HUGGINGFACE, "float16"),  # 12GB -> fp16
            (24.0, True, EngineKind.HUGGINGFACE, "float16"),  # L4/A100 -> HF fp16
        ],
    )
    def test_auto_select_by_vram(
        self, vram_gb: float, cuda: bool, expected_kind: EngineKind, expected_compute: str
    ) -> None:
        info = GPUInfo(name="X", vram_gb=vram_gb, cuda_available=cuda, bf16_supported=False)
        spec = EngineRouter.select(info)
        assert spec.kind is expected_kind
        assert spec.compute_type == expected_compute

    def test_user_override_forces_faster_whisper_even_on_big_gpu(self) -> None:
        info = GPUInfo(name="A100", vram_gb=40.0, cuda_available=True, bf16_supported=True)
        spec = EngineRouter.select(info, prefer=EngineKind.FASTER_WHISPER)
        assert spec.kind is EngineKind.FASTER_WHISPER

    def test_user_override_hf_on_tiny_gpu_raises(self) -> None:
        """Below 3.5 GB even the aggressive HF low-memory mode OOMs; strict=True
        must refuse instead of silently downgrading to Faster-Whisper."""
        from taigi_asr.errors import InsufficientVRAMError

        info = GPUInfo(name="MX150", vram_gb=2.0, cuda_available=True, bf16_supported=False)
        with pytest.raises(InsufficientVRAMError):
            EngineRouter.select(info, prefer=EngineKind.HUGGINGFACE, strict=True)

    def test_user_override_hf_on_4gb_succeeds(self) -> None:
        """RTX 3050 Laptop 4 GB (Optimus, full dGPU VRAM available) must be
        able to run HF pipeline in aggressive low-memory mode."""
        info = GPUInfo(name="RTX 3050", vram_gb=4.0, cuda_available=True, bf16_supported=False)
        spec = EngineRouter.select(info, prefer=EngineKind.HUGGINGFACE, strict=True)
        assert spec.kind is EngineKind.HUGGINGFACE
        assert spec.batch_size == 1  # conservative for 4 GB

    def test_batch_size_scales_down_for_small_vram(self) -> None:
        small = GPUInfo(name="3050", vram_gb=4.0, cuda_available=True, bf16_supported=False)
        big = GPUInfo(name="A100", vram_gb=40.0, cuda_available=True, bf16_supported=True)
        assert EngineRouter.select(small).batch_size <= 4
        assert EngineRouter.select(big).batch_size >= 16


class TestEngineSpec:
    def test_is_frozen_dataclass(self) -> None:
        spec = EngineSpec(
            kind=EngineKind.FASTER_WHISPER,
            device="cuda",
            compute_type="int8_float16",
            batch_size=4,
        )
        with pytest.raises((AttributeError, Exception)):
            spec.batch_size = 99  # type: ignore[misc]
