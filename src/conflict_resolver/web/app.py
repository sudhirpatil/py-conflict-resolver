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
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
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


@app.get("/python-versions")
async def python_versions_status():
    """Return availability of each configured Python version."""
    import shutil
    import subprocess
    from conflict_resolver.config import load_config
    cfg = load_config()
    result = {}
    for ver, path in cfg.agent.python_versions.items():
        exe = path.strip() if path.strip() else shutil.which(f"python{ver}")
        if exe and Path(exe).exists():
            try:
                out = subprocess.run(
                    [exe, "--version"], capture_output=True, text=True, timeout=3
                )
                result[ver] = {
                    "available": True,
                    "path": exe,
                    "version": out.stdout.strip() or out.stderr.strip(),
                }
            except Exception:
                result[ver] = {"available": False, "path": exe, "version": ""}
        else:
            result[ver] = {"available": False, "path": exe or "", "version": ""}
    return JSONResponse(result)


@app.post("/resolve")
async def resolve(
    file: UploadFile = File(...),
    python_version: str = Form("3.11"),
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
        from conflict_resolver.agent import ConflictResolverAgent, ManualAnalysisAgent
        from conflict_resolver.config import load_config, LLMConfig, AgentConfig, AppConfig
        from conflict_resolver.llm_factory import create_llm
        from conflict_resolver.venv_manager import VenvManager

        # --- Queue-based logging handler so we can stream log lines to SSE ---
        # Attach directly to the conflict_resolver package logger so uvicorn's
        # root-logger configuration doesn't interfere.
        class QueueHandler(logging.Handler):
            def emit(self, record: logging.LogRecord):
                log_queue.put(json.dumps({"type": "log", "text": self.format(record)}))

        _fmt = logging.Formatter("%(levelname)s %(name)s: %(message)s")

        handler = QueueHandler()
        handler.setFormatter(_fmt)
        handler.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(_fmt)
        console_handler.setLevel(logging.INFO)

        pkg_log = logging.getLogger("conflict_resolver")
        pkg_log.addHandler(handler)
        pkg_log.addHandler(console_handler)
        pkg_log.setLevel(logging.INFO)
        pkg_log.propagate = False  # don't double-log through uvicorn's root handlers

        final_payload: dict[str, Any] = {}

        try:
            # Load settings from config.toml
            cfg = load_config()
            logger.info(
                "Using config: provider=%s model=%s max_loops=%d pip_timeout=%d",
                cfg.llm.provider, cfg.llm.model,
                cfg.agent.max_loops, cfg.agent.pip_timeout,
            )

            llm = create_llm(cfg.llm)

            # Resolve Python interpreter for the requested version
            python_exe: str | None = None
            if python_version in cfg.agent.python_versions:
                configured_path = cfg.agent.python_versions[python_version].strip()
                python_exe = configured_path if configured_path else f"python{python_version}"
                logger.info(
                    "Using Python version: %s (%s)", python_version, python_exe
                )
            else:
                logger.warning(
                    "Requested python_version %r not in config — using system default",
                    python_version,
                )

            # Write uploaded requirements to a temp file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", prefix="uploaded_req_", delete=False
            ) as f:
                f.write(original_text)
                req_path = Path(f.name)

            def pip_callback(line: str):
                log_queue.put(json.dumps({"type": "log", "text": f"pip: {line}"}))

            with VenvManager(
                python=python_exe,
                line_callback=pip_callback,
            ) as vm:
                agent = ConflictResolverAgent(
                    venv_manager=vm,
                    max_loops=cfg.agent.max_loops,
                    pip_timeout=cfg.agent.pip_timeout,
                    pypi_lookup_enabled=cfg.agent.pypi_lookup_enabled,
                    llm=llm,
                )
                result = asyncio.run(agent.resolve(original_text))

            req_path.unlink(missing_ok=True)

            # ── Manual fix recommendations (fully independent from auto-fix) ──
            # 1. Install the user's ORIGINAL requirements.txt in a fresh venv
            # 2. Capture pip output (errors, warnings, successes)
            # 3. Two LLM calls: analysis (summary + root cause), then recommendations
            # 4. Venv is always deleted regardless of outcome
            logger.info("Starting manual fix analysis using original requirements.txt…")
            manual_fix_json = "{}"
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".txt", prefix="manual_req_", delete=False
                ) as mf:
                    mf.write(original_text)
                    manual_req_path = Path(mf.name)

                manual_pip_lines: list[str] = []

                def _manual_pip_callback(line: str):
                    manual_pip_lines.append(line)

                with VenvManager(
                    python=python_exe,
                    line_callback=_manual_pip_callback,
                ) as manual_vm:
                    logger.info("Installing original requirements.txt for manual analysis…")
                    manual_result = manual_vm.install_from_file(
                        manual_req_path, timeout=cfg.agent.pip_timeout
                    )

                manual_req_path.unlink(missing_ok=True)
                manual_pip_output = manual_result.combined_output

                manual_agent = ManualAnalysisAgent(llm=llm)

                logger.info("Generating issue summary and root cause…")
                analysis = asyncio.run(
                    manual_agent.analyze_issue(requirements=original_text, pip_output=manual_pip_output)
                )

                logger.info("Generating manual fix recommendations…")
                recommendations = asyncio.run(
                    manual_agent.recommend_fixes(
                        requirements=original_text,
                        pip_output=manual_pip_output,
                        root_cause=analysis.root_cause,
                    )
                )

                manual_fix_json = json.dumps({
                    "issue_summary": analysis.issue_summary,
                    "root_cause": analysis.root_cause,
                    "recommendations": [r.model_dump() for r in recommendations],
                })
                logger.info("Manual fix analysis complete.")

            except Exception as exc:
                logger.warning("Manual fix analysis failed: %s", exc)
                manual_fix_json = json.dumps({
                    "issue_summary": "Could not generate manual fix recommendations.",
                    "root_cause": str(exc),
                    "recommendations": [],
                })

            if result.success and result.resolved_requirements:
                resolved = result.resolved_requirements
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
                    "manual_fix": manual_fix_json,
                }
            else:
                # Summarise what was tried
                attempts = result.failed_attempts
                summary_lines = []
                for i, fa in enumerate(attempts, 1):
                    req_diff = fa.get("requirements", "(no diff)")
                    summary_lines.append(f"Attempt {i}:\n{req_diff}")

                last_pip = result.last_pip_output
                # Extract error lines for display
                error_lines = [
                    ln for ln in last_pip.splitlines()
                    if any(k in ln.lower() for k in ("error", "conflict", "cannot install",
                                                      "no matching distribution"))
                ]

                partial = result.partial_install_result or {}
                if partial:
                    # Attach original + compatible as plain text for UI diff & download
                    partial = {
                        **partial,
                        "original_requirements": original_text,
                        "compatible_requirements": "\n".join(partial.get("compatible", [])),
                    }
                final_payload = {
                    "type": "result",
                    "success": False,
                    "error_message": result.error_message or "Unknown error",
                    "last_pip_errors": "\n".join(error_lines) or last_pip[-2000:],
                    "attempts_summary": "\n\n".join(summary_lines) or "No attempts recorded.",
                    "manual_fix": manual_fix_json,
                    "partial_install": json.dumps(partial) if partial else "{}",
                }

        except Exception as exc:
            logger.exception("Agent run failed: %s", exc)
            final_payload = {
                "type": "result",
                "success": False,
                "error_message": str(exc),
                "last_pip_errors": "",
                "attempts_summary": "",
                "manual_fix": "{}",
            }
        finally:
            pkg_log.removeHandler(handler)
            pkg_log.removeHandler(console_handler)
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
    uvicorn.run("conflict_resolver.web.app:app", host="0.0.0.0", port=8999, reload=True)


if __name__ == "__main__":
    run()
