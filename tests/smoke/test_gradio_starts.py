"""Smoke test: build the Gradio UI without launching a server."""

from __future__ import annotations

import pytest

try:
    import gradio as gr  # noqa: F401

    _GRADIO_OK = True
except ImportError:  # pragma: no cover
    _GRADIO_OK = False


@pytest.mark.skipif(not _GRADIO_OK, reason="gradio not installed")
def test_build_ui_returns_blocks() -> None:
    from taigi_asr.ui.gradio_app import build_ui

    demo = build_ui()
    # The UI is a gradio.Blocks; we don't launch — just assert structure.
    import gradio as gr

    assert isinstance(demo, gr.Blocks)


@pytest.mark.skipif(not _GRADIO_OK, reason="gradio not installed")
def test_launcher_main_has_argparse(monkeypatch) -> None:
    """Passing --help should exit 0 without blocking."""
    import sys

    from taigi_asr.ui.launcher import main

    monkeypatch.setattr(sys, "argv", ["taigi-asr-ui", "--help"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0
