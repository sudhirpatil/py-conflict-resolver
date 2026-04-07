"""Virtual environment lifecycle management."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import venv
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

    def __init__(self, base_dir: Path | None = None) -> None:
        if base_dir is not None:
            self.venv_path = Path(base_dir)
        else:
            self._tmpdir = tempfile.mkdtemp(prefix="conflict_resolver_venv_")
            self.venv_path = Path(self._tmpdir) / "venv"
        self._created = False

    def __enter__(self) -> "VenvManager":
        self.create()
        return self

    def __exit__(self, *_) -> None:
        self.destroy()

    def create(self) -> None:
        logger.info("Creating virtual environment at %s", self.venv_path)
        try:
            venv.create(str(self.venv_path), with_pip=True, clear=True)
        except Exception:
            # Fallback: use subprocess
            subprocess.run(
                [sys.executable, "-m", "venv", str(self.venv_path)],
                check=True,
                capture_output=True,
                text=True,
            )
        self._created = True
        logger.debug("Virtual environment created")

    @property
    def pip_path(self) -> Path:
        if sys.platform == "win32":
            return self.venv_path / "Scripts" / "pip.exe"
        return self.venv_path / "bin" / "pip"

    def install_from_file(self, requirements_path: Path, timeout: int = 120) -> InstallResult:
        logger.info("Running pip install -r %s", requirements_path)
        env = {**os.environ, "PIP_NO_COLOR": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
        try:
            result = subprocess.run(
                [str(self.pip_path), "install", "-r", str(requirements_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
        except subprocess.TimeoutExpired:
            logger.warning("pip install timed out after %ds", timeout)
            return InstallResult(
                success=False,
                returncode=-1,
                stdout="",
                stderr=f"pip install timed out after {timeout} seconds",
                combined_output=f"TIMEOUT: pip install timed out after {timeout} seconds",
            )

        combined = f"=== pip stdout ===\n{result.stdout}\n\n=== pip stderr ===\n{result.stderr}"
        success = result.returncode == 0
        if success:
            logger.info("pip install succeeded")
        else:
            logger.warning("pip install failed (exit code %d)", result.returncode)
            logger.debug("pip output:\n%s", combined)

        return InstallResult(
            success=success,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            combined_output=combined,
        )

    def destroy(self) -> None:
        if hasattr(self, "_tmpdir") and Path(self._tmpdir).exists():
            logger.info("Deleting virtual environment at %s", self._tmpdir)
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        elif self.venv_path.exists():
            logger.info("Deleting virtual environment at %s", self.venv_path)
            shutil.rmtree(self.venv_path, ignore_errors=True)
