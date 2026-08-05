"""Virtual environment lifecycle management."""

from __future__ import annotations

import logging
import os
import re
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
        self._bootstrap_packages: set[str] = set()

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

        # Snapshot the bootstrap package set (pip/setuptools/wheel and whatever
        # they pull in, e.g. packaging) so freeze() can exclude it later — these
        # aren't dependencies of whatever requirements get installed afterward.
        self._bootstrap_packages = self._installed_names()

        self._created = True
        logger.debug("Virtual environment created")

    def _installed_names(self, timeout: int = 30) -> set[str]:
        """Return the lowercase names of all packages currently installed via `pip freeze`."""
        env = {**os.environ, "PIP_NO_COLOR": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
        try:
            result = subprocess.run(
                [str(self.pip_path), "freeze"],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
        except Exception as exc:
            logger.warning("pip freeze (bootstrap snapshot) failed: %s", exc)
            return set()

        names = set()
        for line in result.stdout.splitlines():
            line = line.strip()
            m = self._FREEZE_NAME_RE.match(line)
            if m:
                names.add(m.group(1).lower())
        return names

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

    def dry_run_install(self, requirements_path: Path, timeout: int = 60) -> tuple[bool, str]:
        """Run pip install --dry-run and return (has_conflicts, combined_output).

        has_conflicts is True when pip's resolver found errors (exit code != 0).
        Returns (False, "") if dry-run itself cannot run (pip < 22.1, timeout, etc.)
        so the caller falls back to a real install.
        """
        env = {**os.environ, "PIP_NO_COLOR": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
        try:
            result = subprocess.run(
                [str(self.pip_path), "install", "--dry-run", "-r", str(requirements_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            combined = f"=== pip dry-run stdout ===\n{result.stdout}\n\n=== pip dry-run stderr ===\n{result.stderr}"
            has_conflicts = result.returncode != 0
            logger.info(
                "pip dry-run completed (exit code %d) — %s",
                result.returncode,
                "conflicts detected" if has_conflicts else "no conflicts",
            )
            return has_conflicts, combined
        except subprocess.TimeoutExpired:
            logger.warning("pip dry-run timed out after %ds — falling back to real install", timeout)
            return False, ""
        except Exception as exc:
            logger.warning("pip dry-run failed: %s — falling back to real install", exc)
            return False, ""

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

    def install_single_no_deps(self, package_spec: str, timeout: int = 60) -> InstallResult:
        """Install one package with --no-deps, bypassing the resolver.

        Used as a last-resort for packages that cannot be reconciled with the
        rest of the requirements set. Returns an InstallResult like install_from_file.
        """
        env = {**os.environ, "PIP_NO_COLOR": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
        logger.info("Force-installing (--no-deps): %s", package_spec)
        try:
            result = subprocess.run(
                [str(self.pip_path), "install", "--no-deps", package_spec],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            combined = f"=== pip install --no-deps {package_spec} stdout ===\n{result.stdout}\n\n=== stderr ===\n{result.stderr}"
            success = result.returncode == 0
            if success:
                logger.info("Force-install succeeded: %s", package_spec)
            else:
                logger.warning("Force-install failed (exit %d): %s", result.returncode, package_spec)
            return InstallResult(
                success=success,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                combined_output=combined,
            )
        except subprocess.TimeoutExpired:
            logger.warning("Force-install timed out after %ds: %s", timeout, package_spec)
            return InstallResult(success=False, returncode=-1, stdout="", stderr="timeout", combined_output=f"TIMEOUT: {package_spec}")
        except Exception as exc:
            logger.error("Force-install error for %s: %s", package_spec, exc)
            return InstallResult(success=False, returncode=-1, stdout="", stderr=str(exc), combined_output=f"ERROR: {exc}")

    _FREEZE_NAME_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==")
    _FREEZE_EXCLUDE = {"pip", "setuptools", "wheel"}

    def freeze(self, timeout: int = 30) -> list[str]:
        """Return `pip freeze` output as sorted "name==version" lines.

        Captures the full set of packages actually installed in the venv —
        pip's resolver has already worked out the transitive closure — excluding
        whatever was already present right after create() (pip/setuptools/wheel
        and anything they pulled in, e.g. packaging), since those aren't
        dependencies of the requirements that were installed afterward.
        """
        env = {**os.environ, "PIP_NO_COLOR": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
        try:
            result = subprocess.run(
                [str(self.pip_path), "freeze"],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
        except subprocess.TimeoutExpired:
            logger.warning("pip freeze timed out after %ds", timeout)
            return []
        except Exception as exc:
            logger.warning("pip freeze failed: %s", exc)
            return []

        excluded = self._FREEZE_EXCLUDE | self._bootstrap_packages
        lines = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = self._FREEZE_NAME_RE.match(line)
            if m and m.group(1).lower() in excluded:
                continue
            lines.append(line)
        return sorted(lines, key=str.lower)

    def run_pip_check(self) -> str:
        """Run `pip check` and return combined output.

        pip check scans installed metadata and reports unmet dependencies
        without re-running the resolver.
        """
        env = {**os.environ, "PIP_NO_COLOR": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
        try:
            result = subprocess.run(
                [str(self.pip_path), "check"],
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
            )
            output = (result.stdout + result.stderr).strip()
            if result.returncode == 0:
                logger.info("pip check: no broken dependencies")
            else:
                logger.warning("pip check: found dependency issues")
                for line in output.splitlines():
                    logger.warning("  %s", line)
            return output or "No issues found."
        except Exception as exc:
            logger.warning("pip check failed: %s", exc)
            return f"pip check could not run: {exc}"

    def destroy(self) -> None:
        if hasattr(self, "_tmpdir") and Path(self._tmpdir).exists():
            logger.info("Deleting virtual environment at %s", self._tmpdir)
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        elif self.venv_path.exists():
            logger.info("Deleting virtual environment at %s", self.venv_path)
            shutil.rmtree(self.venv_path, ignore_errors=True)
