"""FastAPI web UI for py-conflict-resolver."""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
import queue
import tempfile
import threading
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

load_dotenv(override=False)

logger = logging.getLogger(__name__)

app = FastAPI(title="py-conflict-resolver")

# Mount static files (index.html)
_STATIC = Path(__file__).parent / "static"
_STATIC.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    return (_STATIC / "index.html").read_text(encoding="utf-8")


@app.post("/resolve")
async def resolve(
    file: UploadFile = File(...),
    provider: str = Form("openai"),
    model: str = Form("gpt-4.1"),
    max_loops: int = Form(10),
    python_version: str = Form(""),
):
    """
    Upload a requirements.txt, run the agent, stream logs via SSE.

    The response is a text/event-stream where each event has a `type` field:
      log      – a single log line from the agent / pip
      result   – final JSON payload (success/failure + diff or summary)
    """
    contents = await file.read()
    original_text = contents.decode("utf-8")

    log_queue: queue.Queue[str | None] = queue.Queue()

    def _run():
        """Runs the blocking agent in a background thread."""
        import sys
        from conflict_resolver.agent import build_graph
        from conflict_resolver.config import load_config, LLMConfig, AgentConfig, AppConfig
        from conflict_resolver.llm_factory import create_llm
        from conflict_resolver.venv_manager import VenvManager

        # --- Queue-based logging handler so we can stream log lines to SSE ---
        # Attach directly to the conflict_resolver package logger so uvicorn's
        # root-logger configuration doesn't interfere.
        class QueueHandler(logging.Handler):
            def emit(self, record: logging.LogRecord):
                log_queue.put(json.dumps({"type": "log", "text": self.format(record)}))

        handler = QueueHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        handler.setLevel(logging.DEBUG)

        pkg_log = logging.getLogger("conflict_resolver")
        pkg_log.addHandler(handler)
        pkg_log.setLevel(logging.DEBUG)
        pkg_log.propagate = False  # don't double-log through uvicorn's root handlers

        final_payload: dict[str, Any] = {}

        try:
            # Build config from form values (skip file-based config)
            llm_cfg = LLMConfig(provider=provider, model=model, temperature=0.2)
            agent_cfg = AgentConfig(max_loops=max_loops, pip_timeout=300)
            available_models: dict[str, list[str]] = {}
            cfg = AppConfig(llm=llm_cfg, agent=agent_cfg, available_models=available_models)

            llm = create_llm(cfg.llm)

            # Write uploaded requirements to a temp file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", prefix="uploaded_req_", delete=False
            ) as f:
                f.write(original_text)
                req_path = Path(f.name)

            def pip_callback(line: str):
                log_queue.put(json.dumps({"type": "log", "text": f"pip: {line}"}))

            with VenvManager(
                python=python_version or None,
                line_callback=pip_callback,
            ) as vm:
                graph = build_graph(llm, vm, max_loops, agent_cfg.pip_timeout)

                initial_state = {
                    "original_requirements_path": str(req_path),
                    "original_requirements": original_text,
                    "current_requirements": original_text,
                    "attempt_count": 0,
                    "last_install_success": False,
                    "last_pip_output": "",
                    "failed_attempts": [],
                    "messages": [],
                    "resolved_requirements": None,
                    "error_message": None,
                }

                final_state = graph.invoke(initial_state)

            req_path.unlink(missing_ok=True)

            if final_state.get("last_install_success") and final_state.get("resolved_requirements"):
                resolved = final_state["resolved_requirements"]
                diff_lines = list(
                    difflib.unified_diff(
                        original_text.splitlines(),
                        resolved.splitlines(),
                        fromfile="requirements.txt (original)",
                        tofile="requirements.txt (resolved)",
                        lineterm="",
                    )
                )
                final_payload = {
                    "type": "result",
                    "success": True,
                    "resolved": resolved,
                    "diff": "\n".join(diff_lines),
                }
            else:
                # Summarise what was tried
                attempts = final_state.get("failed_attempts", [])
                summary_lines = []
                for i, fa in enumerate(attempts, 1):
                    req_diff = fa.get("requirements", "(no diff)")
                    summary_lines.append(f"Attempt {i}:\n{req_diff}")

                last_pip = final_state.get("last_pip_output", "")
                # Extract error lines for display
                error_lines = [
                    ln for ln in last_pip.splitlines()
                    if any(k in ln.lower() for k in ("error", "conflict", "cannot install",
                                                      "no matching distribution"))
                ]

                final_payload = {
                    "type": "result",
                    "success": False,
                    "error_message": final_state.get("error_message", "Unknown error"),
                    "last_pip_errors": "\n".join(error_lines) or last_pip[-2000:],
                    "attempts_summary": "\n\n".join(summary_lines) or "No attempts recorded.",
                }

        except Exception as exc:
            logger.exception("Agent run failed: %s", exc)
            final_payload = {
                "type": "result",
                "success": False,
                "error_message": str(exc),
                "last_pip_errors": "",
                "attempts_summary": "",
            }
        finally:
            pkg_log.removeHandler(handler)
            # Push final result then sentinel
            log_queue.put(json.dumps(final_payload))
            log_queue.put(None)  # sentinel → stream ends

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    async def event_stream():
        loop = asyncio.get_event_loop()
        while True:
            # Poll queue without blocking the event loop
            try:
                item = await loop.run_in_executor(None, lambda: log_queue.get(timeout=0.1))
            except queue.Empty:
                await asyncio.sleep(0.05)
                continue
            if item is None:
                yield "event: done\ndata: {}\n\n"
                break
            yield f"data: {item}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def run():
    import uvicorn
    uvicorn.run("conflict_resolver.web.app:app", host="0.0.0.0", port=8999, reload=False)


if __name__ == "__main__":
    run()
