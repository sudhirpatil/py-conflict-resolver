"""Virtual environment lifecycle management."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class InstallResult:
    success: bool
    returncode: int
    stdout: str
    stderr: str
    combined_output: str


class VenvManager:
    """Context manager that creates a temporary venv, provides pip install,
    and always deletes the venv on exit."""

    def __init__(
        self,
        base_dir: Path | None = None,
        python: str | None = None,
        line_callback=None,
    ) -> None:
        if base_dir is not None:
            self.venv_path = Path(base_dir)
        else:
            self._tmpdir = tempfile.mkdtemp(prefix="conflict_resolver_venv_")
            self.venv_path = Path(self._tmpdir) / "venv"
        self._python = python  # None means use the current interpreter
        self._line_callback = line_callback  # Callable[[str], None] | None
        self._created = False

    def __enter__(self) -> "VenvManager":
        self.create()
        return self

    def __exit__(self, *_) -> None:
        self.destroy()

    def create(self) -> None:
        python_exe = self._resolve_python()
        logger.info(
            "Creating virtual environment at %s (python: %s)",
            self.venv_path,
            python_exe,
        )
        # Always use subprocess so we can specify the Python executable
        result = subprocess.run(
            [python_exe, "-m", "venv", str(self.venv_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to create virtual environment with {python_exe!r}:\n"
                f"{result.stderr.strip()}"
            )

        # Ensure build tools are available so pip can build sdist packages
        # (e.g. older numpy that use setuptools.build_meta)
        logger.info("Installing build tools (setuptools, wheel) into venv…")
        build_result = subprocess.run(
            [str(self.pip_path), "install", "--quiet", "setuptools", "wheel"],
            capture_output=True,
            text=True,
        )
        if build_result.returncode != 0:
            logger.warning("Failed to install build tools: %s", build_result.stderr.strip())

        self._created = True
        logger.debug("Virtual environment created")

    def _resolve_python(self) -> str:
        """Return the Python executable to use for venv creation.

        Accepts:
          - None / ""         → current interpreter (sys.executable)
          - "3.11"            → looks for python3.11 on PATH
          - "python3.11"      → used as-is
          - "/usr/bin/python3" → absolute path, used as-is
        """
        if not self._python:
            return sys.executable

        spec = self._python.strip()

        # Bare version like "3.11" or "3.11.2" → expand to "python3.11"
        if spec and spec[0].isdigit():
            # Use only major.minor for the executable name
            parts = spec.split(".")
            spec = f"python{parts[0]}.{parts[1]}" if len(parts) >= 2 else f"python{parts[0]}"

        # Absolute path — validate it exists
        if spec.startswith("/") or spec.startswith("\\"):
            if not Path(spec).exists():
                raise FileNotFoundError(f"Python executable not found: {spec}")
            return spec

        # Name on PATH — verify it can be found
        found = shutil.which(spec)
        if found is None:
            raise FileNotFoundError(
                f"Python executable {spec!r} not found on PATH. "
                f"Install Python {self._python} and ensure it is on your PATH."
            )
        return found

    @property
    def pip_path(self) -> Path:
        if sys.platform == "win32":
            return self.venv_path / "Scripts" / "pip.exe"
        return self.venv_path / "bin" / "pip"

    def install_from_file(self, requirements_path: Path, timeout: int = 120) -> InstallResult:
        logger.info("Running pip install -r %s", requirements_path)
        # PIP_NO_COLOR kept for the captured copy sent to the LLM; real-time
        # output goes to sys.stdout/stderr directly so the user sees progress.
        env = {**os.environ, "PIP_NO_COLOR": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"}

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        try:
            process = subprocess.Popen(
                [str(self.pip_path), "install", "-r", str(requirements_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )

            # Stream lines from pipe, capture for LLM, and either:
            #   - print to terminal (CLI mode, no callback), or
            #   - forward to callback (web mode) — not both, to keep server logs clean
            def _stream(pipe, store: list[str], file):
                for line in pipe:
                    store.append(line)
                    if self._line_callback:
                        self._line_callback(line.rstrip())
                    else:
                        print(line, end="", file=file, flush=True)

            t_out = threading.Thread(target=_stream, args=(process.stdout, stdout_lines, sys.stdout))
            t_err = threading.Thread(target=_stream, args=(process.stderr, stderr_lines, sys.stderr))
            t_out.start()
            t_err.start()

            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                t_out.join()
                t_err.join()
                logger.warning("pip install timed out after %ds", timeout)
                return InstallResult(
                    success=False,
                    returncode=-1,
                    stdout="".join(stdout_lines),
                    stderr=f"pip install timed out after {timeout} seconds",
                    combined_output=f"TIMEOUT: pip install timed out after {timeout} seconds",
                )

            t_out.join()
            t_err.join()

        except Exception as exc:
            logger.error("Failed to launch pip: %s", exc)
            return InstallResult(
                success=False,
                returncode=-1,
                stdout="",
                stderr=str(exc),
                combined_output=f"ERROR: {exc}",
            )

        stdout_text = "".join(stdout_lines)
        stderr_text = "".join(stderr_lines)
        combined = f"=== pip stdout ===\n{stdout_text}\n\n=== pip stderr ===\n{stderr_text}"
        success = process.returncode == 0

        if success:
            logger.info("pip install succeeded")
        else:
            logger.warning("pip install failed (exit code %d)", process.returncode)

        return InstallResult(
            success=success,
            returncode=process.returncode,
            stdout=stdout_text,
            stderr=stderr_text,
            combined_output=combined,
        )

    def destroy(self) -> None:
        if hasattr(self, "_tmpdir") and Path(self._tmpdir).exists():
            logger.info("Deleting virtual environment at %s", self._tmpdir)
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        elif self.venv_path.exists():
            logger.info("Deleting virtual environment at %s", self.venv_path)
            shutil.rmtree(self.venv_path, ignore_errors=True)
