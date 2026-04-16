# FAQ / Troubleshooting

## Install

### "CUDA not available" on a machine with an NVIDIA GPU

`torch.cuda.is_available()` returns False when the installed torch wheel is CPU-only. Reinstall the CUDA wheel:

```bash
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
# Verify:
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> The repo's `install.{sh,bat}` and the `Dockerfile` all target CUDA 12.1 wheels
> for consistency. If your driver only supports a different CUDA minor version,
> pick the matching wheel from <https://pytorch.org/get-started/locally/>.

If `nvidia-smi` works but torch still says no CUDA:
- Check NVIDIA driver version >= 525 (CUDA 12.x requires modern driver).
- On WSL2: run `wsl --update` and install the Windows NVIDIA driver (not a separate Linux driver inside WSL).

### Model download failed / huggingface_hub timeout

```bash
# Set mirror (China users):
export HF_ENDPOINT=https://hf-mirror.com

# Or retry the preload:
python -c "from taigi_asr.engines.faster_whisper import FasterWhisperEngine; FasterWhisperEngine.preload()"
```

### `bitsandbytes` install fails on Windows

Expected. bitsandbytes does not support Windows natively. The router only routes to the `int8` (bitsandbytes) HF path on Linux. On Windows with a 4-8 GB GPU you automatically get Faster-Whisper int8_float16 instead — which on RTX 3050 is actually **faster** than HF int8 due to Ampere Tensor Core kernels.

If you really want HF int8 on Windows, use the Docker path:

```bash
docker compose up -d
```

---

## Runtime

### Out-of-memory on RTX 3050 4GB

By default the router picks `int8_float16` + `batch=4` on 4 GB cards which peaks around 2.9 GB. OOM usually means:
1. **Other apps hogging VRAM.** Close Chrome hardware acceleration, OBS, games, VSCode with GPU rendering. Run `nvidia-smi` to confirm.
2. **Forced `--engine hf` override.** The HF path peaks higher than Faster-Whisper on a 4 GB card; on Windows (no bitsandbytes int8) it ends up at `fp16` + `batch=1` which is still risky. Drop the override and let auto-routing pick Faster-Whisper, or close other GPU apps to free headroom.
3. **Huge beam_size (Faster-Whisper only).** `--beam-size 12` is the documented ceiling; 15+ WILL OOM. This flag is silently ignored by the HuggingFace engine.

Emergency recovery: set env before launching:
```
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   (Windows)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (Linux)
```
(The package already sets this at import time, but if you set it **before** launching Python you get it even earlier in the CUDA context.)

### "torch.compile" fails or warns

`torch.compile` is only attempted by the HuggingFace engine, and failures are caught + logged as INFO (`torch.compile unavailable; using eager mode`). Eager mode is fully functional, just slower by ~15%. Known triggers:
- Windows PyTorch 2.x: inductor backend historically flaky.
- Python 3.13: dynamo coverage incomplete.
- Triton not installed: compile falls through to eager.

No action required — the engine falls back gracefully.

### Transcription is slow on GPU

Expected real-time factor (xRT) on RTX 3050 Laptop 4GB with `int8_float16` + `beam=5` + `batch=4`:

- **Short audio (< 30s)**: dominated by 6-9s model load. xRT looks poor but total is tiny.
- **Long audio (> 5 min)**: ~3-5x real-time. 1 hour of audio takes ~12-20 min.

If you see worse numbers:
- Confirm `nvidia-smi` shows 90%+ GPU util during transcribe.
- Verify `compute_type` is `int8_float16`, not `float16` (fp16 is slower on Ampere consumer cards for this workload).
- Check `batch_size`. For long audio, `batch=8` on 4 GB is OK but risky if other apps use VRAM.
- Ensure ffmpeg is on PATH so audio preprocessing is native speed (pydub fallback uses Python decode loop — much slower).

### WSL2: GPU not detected inside container

1. Confirm driver:
   ```
   nvidia-smi       (inside WSL2, should list the GPU)
   ```
   If not listed, install the latest Windows NVIDIA driver and run `wsl --update`.
2. Install NVIDIA Container Toolkit inside WSL2:
   ```bash
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```
3. Smoke test:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.1-runtime-ubuntu22.04 nvidia-smi
   ```

---

## Accuracy

### Transcript has wrong characters

Breeze-ASR-26 outputs **Mandarin Chinese characters** phonetically mapped from Taiwanese Hokkien speech. Expect:
- Similar-sounding characters may swap (`醫` vs `他` — both "i"/"ti" in Hokkien).
- Rare terms or idioms (`玲瓏` a.k.a. "liàn-lóng") may be substituted with common homophones.
- Background music, multiple speakers, dialect mixing degrade accuracy substantially.

Tuning to try:
```bash
taigi-asr audio.m4a --beam-size 10 --best-of 10
```

Bigger beam + best_of helps borderline cases but does not recover fundamentally-misrecognized words.

### Timestamps are off

Faster-Whisper uses segment-level timestamps from the decoder. Known caveats:
- VAD (default on) pads silence by 200 ms — subtract that if aligning to a waveform.
- Very short audio (< 2 s) sometimes yields `0.0 -> 0.0` for the only segment; the formatter clamps to `start + 1.0` seconds.
- `--word-timestamps` on Faster-Whisper runs an additional CTC alignment pass; slower but more precise per-word boundaries.
- `--word-timestamps` on the HuggingFace engine is forwarded to the Transformers pipeline as `return_timestamps="word"`; quality depends on the model's decoder and can be noisier than Faster-Whisper's forced alignment.
