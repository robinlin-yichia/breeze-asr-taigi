"""Entry point that launches the Gradio UI and auto-opens the browser."""

from __future__ import annotations

import argparse
import logging
import threading
import webbrowser

from taigi_asr.config import GRADIO_HOST, GRADIO_PORT
from taigi_asr.ui import gradio_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Taigi ASR Gradio UI")
    parser.add_argument("--host", default=GRADIO_HOST, help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=GRADIO_PORT, help="Port (default: 7860)")
    parser.add_argument("--no-browser", action="store_true", help="Skip auto-opening browser")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    parser.add_argument("--server-name", help=argparse.SUPPRESS)  # Docker compat
    args = parser.parse_args()

    # Docker sets --server-name 0.0.0.0
    host = args.server_name or args.host

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    ui_module_theme = getattr(gradio_app, "THEME", None)
    demo = gradio_app.build_ui()

    if not args.no_browser and host in ("127.0.0.1", "localhost"):
        threading.Timer(1.5, lambda: webbrowser.open(f"http://{host}:{args.port}")).start()

    import gradio as gr

    launch_kwargs: dict = {
        "server_name": host,
        "server_port": args.port,
        "share": args.share,
        "inbrowser": False,
        "show_error": True,
    }
    # Gradio 6 wants theme on launch(); 4.x wants it on Blocks. We pass here
    # and silently drop the arg on 4.x via the inspect check.
    try:
        import inspect

        if "theme" in inspect.signature(demo.launch).parameters:
            launch_kwargs["theme"] = ui_module_theme or gr.themes.Soft()
    except Exception:  # pragma: no cover
        pass
    demo.queue().launch(**launch_kwargs)


if __name__ == "__main__":
    main()
