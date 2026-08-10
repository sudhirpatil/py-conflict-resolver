"""NOOA-based agent for the conflict-resolution loop.

Replaces the previous LangGraph StateGraph: the install → analyze → fix retry
loop is now plain Python control flow in ConflictResolverAgent.resolve(), with
LLM calls made through single-shot, structured-output generation methods
(PredictStrategy) rather than a graph of nodes and a router.
"""

from __future__ import annotations

import logging
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nooa import Agent
from nooa.decorators import strategy
from nooa.strategies import PredictStrategy
from pydantic import BaseModel

from conflict_resolver.prompts import condense_pip_output, diff_requirements
from conflict_resolver.req_tools import (
    _REQ_LINE_RE,
    AddPackage,
    FixAction,
    SetPackageVersion,
    apply_fix_actions,
)
from conflict_resolver.venv_manager import InstallResult, VenvManager

logger = logging.getLogger(__name__)

# ─── Result / support types ────────────────────────────────────────────────


@dataclass
class ResolveResult:
    success: bool
    resolved_requirements: str | None = None
    error_message: str | None = None
    attempt_count: int = 0
    last_pip_output: str = ""
    failed_attempts: list[dict[str, str]] = field(default_factory=list)
    partial_install_result: dict[str, Any] | None = None


class Partition(BaseModel):
    """Split of requirements into a compatible group and a conflicting group."""

    compatible: list[str]
    conflicting: list[str]
    reason: str


# ─── Deterministic helpers (module-level, no LLM) ──────────────────────────


def _packages_from_requirements(requirements_text: str) -> list[str]:
    """Return package names parsed from a requirements text block."""
    names = []
    for line in requirements_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        m = _REQ_LINE_RE.match(stripped)
        if m:
            names.append(m.group(1))
    return names


_EXACT_PIN_RE = re.compile(r"^==\s*(.+)$")


def _pinned_versions_from_requirements(requirements_text: str) -> dict[str, str]:
    """Return {package: version} for lines with an exact pin (==x.y.z only).

    Only exact pins are returned because requires_dist must be fetched for a
    specific version — range constraints don't map to a single PyPI endpoint.
    """
    result: dict[str, str] = {}
    for line in requirements_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        m = _REQ_LINE_RE.match(stripped)
        if not m:
            continue
        pkg = m.group(1)
        spec = (m.group(2) or "").strip()
        em = _EXACT_PIN_RE.match(spec)
        if em:
            result[pkg] = em.group(1).strip()
    return result


# ─── Agent ──────────────────────────────────────────────────────────────────


class ConflictResolverAgent(Agent):
    """You are an expert Python packaging engineer specializing in resolving pip dependency conflicts."""

    def __init__(
        self,
        venv_manager: VenvManager,
        max_loops: int,
        pip_timeout: int = 300,
        pypi_lookup_enabled: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.venv_manager = venv_manager
        self.max_loops = max_loops
        self.pip_timeout = pip_timeout
        self.pypi_lookup_enabled = pypi_lookup_enabled

        self.original_requirements = ""
        self.current_requirements = ""
        self.attempt_count = 0
        self.last_install_success = False
        self.last_pip_output = ""
        self.last_dry_run_output = ""
        self.resolved_requirements: str | None = None
        self.failed_attempts: list[dict[str, str]] = []
        self.pypi_versions: dict[str, list[str]] = {}
        self.pypi_requires_dist: dict[str, list[str]] = {}

    # ── Orchestrator (plain Python, not LLM-driven) ────────────────────────

    async def resolve(self, original_requirements: str) -> ResolveResult:
        """Run the install → analyze → fix loop until success or max_loops, then fall back."""
        self.original_requirements = original_requirements
        self.current_requirements = original_requirements

        for attempt in range(1, self.max_loops + 1):
            self._install(attempt)

            if self.last_install_success:
                logger.info("Resolved successfully after %d attempt(s)", self.attempt_count)
                return ResolveResult(
                    success=True,
                    resolved_requirements=self.resolved_requirements,
                    attempt_count=self.attempt_count,
                    last_pip_output=self.last_pip_output,
                    failed_attempts=self.failed_attempts,
                )

            if self.pypi_lookup_enabled:
                self._pypi_lookup()

            pip_output_condensed = condense_pip_output(self.last_pip_output)
            if self.failed_attempts:
                logger.info(
                    "Including %d previous failed attempt(s) in LLM context to avoid repeats",
                    len(self.failed_attempts),
                )

            logger.info("─" * 60)
            logger.info("Analyzing pip failure with LLM (attempt %d)", attempt)
            logger.info("─" * 60)

            actions = await self.propose_fix(
                attempt=attempt,
                max_loops=self.max_loops,
                current_requirements=self.current_requirements,
                pip_output=pip_output_condensed,
                dry_run_output=self.last_dry_run_output,
                pypi_versions={pkg: versions[:10] for pkg, versions in self.pypi_versions.items()},
                pypi_requires_dist=self.pypi_requires_dist,
                failed_attempts=[
                    {"requirements": fa["requirements"], "pip_output": fa["pip_output"]}
                    for fa in self.failed_attempts
                ],
            )

            self._record_failed_attempt(pip_output_condensed)
            self._apply_fix(actions)

        partial = await self._partition_fallback()
        msg = (
            f"Could not resolve conflicts after {self.attempt_count} attempt(s) "
            f"(max_loops={self.max_loops})."
        )
        logger.error(msg)
        return ResolveResult(
            success=False,
            error_message=msg,
            attempt_count=self.attempt_count,
            last_pip_output=self.last_pip_output,
            failed_attempts=self.failed_attempts,
            partial_install_result=partial,
        )

    # ── Deterministic helpers (ordinary Python, no LLM) ─────────────────────

    def _install(self, attempt: int) -> None:
        """Run pip dry-run then a real install; capture the transitive closure on success."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", prefix="requirements_", delete=False
        ) as f:
            f.write(self.current_requirements)
            req_file = Path(f.name)

        try:
            logger.info("Install attempt %d", attempt)
            logger.info("Running pip dry-run to detect conflicts before full install…")
            has_conflicts, dry_run_output = self.venv_manager.dry_run_install(req_file)

            if has_conflicts:
                logger.info(
                    "Conflicts detected by dry-run — skipping real pip install to save time"
                )
                result = InstallResult(
                    success=False,
                    returncode=1,
                    stdout="",
                    stderr="",
                    combined_output=dry_run_output,
                )
            else:
                result = self.venv_manager.install_from_file(req_file, timeout=self.pip_timeout)
        finally:
            req_file.unlink(missing_ok=True)

        self.attempt_count = attempt
        self.last_install_success = result.success
        self.last_pip_output = result.combined_output
        self.last_dry_run_output = dry_run_output
        self.resolved_requirements = None

        if result.success:
            frozen = self.venv_manager.freeze()
            if frozen:
                self.resolved_requirements = "\n".join(frozen) + "\n"
                logger.info(
                    "Captured %d installed package(s) via pip freeze (includes transitive deps)",
                    len(frozen),
                )
            else:
                logger.warning("pip freeze returned no output — using top-level requirements only")
                self.resolved_requirements = self.current_requirements

    def _pypi_lookup(self) -> None:
        from conflict_resolver.pypi_client import fetch_requires_dist_bulk, fetch_versions_bulk

        packages = _packages_from_requirements(self.current_requirements)
        to_fetch = [p for p in packages if p not in self.pypi_versions]
        if to_fetch:
            logger.info("Fetching PyPI version data for %d package(s)…", len(to_fetch))
            fresh = fetch_versions_bulk(to_fetch)
            logger.info(
                "PyPI version lookup complete: %d/%d packages found", len(fresh), len(to_fetch)
            )
            self.pypi_versions.update(fresh)
        else:
            logger.info("All packages already in PyPI versions cache — skipping")

        pinned = _pinned_versions_from_requirements(self.current_requirements)
        to_fetch_deps = {pkg: ver for pkg, ver in pinned.items() if pkg not in self.pypi_requires_dist}
        if to_fetch_deps:
            logger.info(
                "Fetching dependency metadata (requires_dist) for %d pinned package(s): %s",
                len(to_fetch_deps),
                ", ".join(f"{p}=={v}" for p, v in to_fetch_deps.items()),
            )
            fresh_deps = fetch_requires_dist_bulk(to_fetch_deps)
            logger.info(
                "requires_dist lookup complete: %d/%d packages have declared dependencies",
                len(fresh_deps),
                len(to_fetch_deps),
            )
            self.pypi_requires_dist.update(fresh_deps)
        else:
            skipped = set(pinned) - set(to_fetch_deps)
            if skipped:
                logger.info(
                    "requires_dist: %d package(s) already cached — skipping fetch (%s)",
                    len(skipped),
                    ", ".join(sorted(skipped)),
                )
            else:
                logger.info(
                    "requires_dist: no exact-pinned packages found in requirements "
                    "(only exact ==x.y.z pins are looked up; range constraints are skipped)"
                )

    def _record_failed_attempt(self, pip_output_condensed: str) -> None:
        prev_requirements = (
            self.failed_attempts[-1]["_full_requirements"]
            if self.failed_attempts
            else self.original_requirements
        )
        self.failed_attempts.append(
            {
                "requirements": diff_requirements(prev_requirements, self.current_requirements),
                "pip_output": pip_output_condensed,
                "_full_requirements": self.current_requirements,
            }
        )

    def _apply_fix(self, actions: list[FixAction]) -> None:
        if not actions:
            logger.warning("LLM proposed no fix actions — retaining previous requirements")
            logger.info("─" * 60)
            return

        logger.info("Applying %d fix action(s) from LLM…", len(actions))
        fixed = apply_fix_actions(self.current_requirements, actions)

        non_empty = [
            line for line in fixed.splitlines() if line.strip() and not line.strip().startswith("#")
        ]
        if not non_empty:
            logger.warning(
                "Fix actions produced an empty requirements block — retaining previous requirements"
            )
            logger.info("─" * 60)
            return

        prev_lines = set(self.current_requirements.splitlines())
        new_lines = set(fixed.splitlines())
        added = new_lines - prev_lines
        removed = prev_lines - new_lines
        if added or removed:
            logger.info("Requirements diff from LLM fix:")
            for line in sorted(removed):
                if line.strip():
                    logger.info("  - %s", line)
            for line in sorted(added):
                if line.strip():
                    logger.info("  + %s", line)
        else:
            logger.info("LLM returned identical requirements (no changes)")

        self.current_requirements = fixed
        logger.info("Proceeding to next install attempt…")
        logger.info("─" * 60)

    async def _partition_fallback(self) -> dict[str, Any]:
        """Last-resort partial install: split, install what installs, force the rest."""
        logger.info("─" * 60)
        logger.info("Partitioning requirements into compatible vs conflicting groups…")
        logger.info("─" * 60)

        partition = await self.partition_requirements(
            current_requirements=self.current_requirements,
            pip_output=self.last_pip_output,
        )
        logger.info(
            "Compatible packages (%d): %s", len(partition.compatible), ", ".join(partition.compatible)
        )
        logger.info(
            "Conflicting packages (%d): %s", len(partition.conflicting), ", ".join(partition.conflicting)
        )
        if partition.reason:
            logger.info("Reason: %s", partition.reason)

        result: dict[str, Any] = {
            "compatible": partition.compatible,
            "conflicting": partition.conflicting,
            "reason": partition.reason,
        }

        if partition.compatible:
            logger.info("Installing %d compatible package(s)…", len(partition.compatible))
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", prefix="compatible_req_", delete=False
            ) as f:
                f.write("\n".join(partition.compatible) + "\n")
                req_file = Path(f.name)
            try:
                install_result = self.venv_manager.install_from_file(req_file, timeout=self.pip_timeout)
            finally:
                req_file.unlink(missing_ok=True)
            result["compatible_install_success"] = install_result.success
            result["compatible_install_output"] = install_result.combined_output
            if install_result.success:
                logger.info("Compatible packages installed successfully")
            else:
                logger.warning("Compatible package install failed — proceeding anyway")
        else:
            logger.info("No compatible packages to install — skipping")

        if partition.conflicting:
            logger.info(
                "Force-installing %d conflicting package(s) with --no-deps…", len(partition.conflicting)
            )
            forced_results: list[dict] = []
            pip_commands: list[str] = []
            for pkg_spec in partition.conflicting:
                cmd = f"pip install --no-deps {pkg_spec}"
                pip_commands.append(cmd)
                logger.info("  %s", cmd)
                forced = self.venv_manager.install_single_no_deps(pkg_spec, timeout=self.pip_timeout)
                forced_results.append(
                    {
                        "package": pkg_spec,
                        "command": cmd,
                        "success": forced.success,
                        "output": forced.combined_output,
                    }
                )
            result["forced_results"] = forced_results
            result["pip_commands"] = pip_commands
        else:
            logger.info("No conflicting packages to force-install — skipping")

        logger.info("Running pip check to identify unmet dependencies…")
        result["pip_check_output"] = self.venv_manager.run_pip_check()

        return result

    # ── Generation methods (LLM-driven, single-shot structured output) ─────

    @strategy(PredictStrategy())
    async def propose_fix(
        self,
        attempt: int,
        max_loops: int,
        current_requirements: str,
        pip_output: str,
        dry_run_output: str,
        pypi_versions: dict[str, list[str]],
        pypi_requires_dist: dict[str, list[str]],
        failed_attempts: list[dict[str, str]],
    ) -> list[SetPackageVersion | AddPackage]:
        """Propose targeted fixes for a pip dependency conflict.

        This is attempt {attempt} of {max_loops}.

        You are given the current requirements.txt contents, pip install errors, and
        real available versions from PyPI for each package. Propose TARGETED changes —
        only touch packages involved in the conflict.

        Rules:
        - Propose changes for ONLY the packages that need changing — leave others untouched.
        - NEVER remove any package — removing packages may break programs that depend on them.
        - Resolve conflicts by adjusting version constraints, not by dropping packages.
        - Pin versions precisely (e.g. ==1.2.3) when resolving conflicts.
        - Use ONLY version strings from the available-versions data below — never invent versions.
        - Never hallucinate package names that do not exist on PyPI.
        - Do NOT repeat any fix already tried in a previous failed attempt below.

        Current requirements.txt:
        {current_requirements}

        Available versions on PyPI (latest 10 shown per package):
        {pypi_versions}

        Declared dependencies of pinned packages (from PyPI metadata):
        {pypi_requires_dist}

        pip dry-run (full dependency conflict graph), if available:
        {dry_run_output}

        pip errors:
        {pip_output}

        Previous failed attempts — do NOT repeat these fixes:
        {failed_attempts}
        """
        ...

    @strategy(PredictStrategy())
    async def partition_requirements(self, current_requirements: str, pip_output: str) -> Partition:
        """Split requirements into a compatible group and a conflicting group.

        The conflicts below could not be fully resolved after repeated fix attempts.
        Split the packages into two groups:

        1. compatible: packages that CAN be installed together without conflicts.
           Keep their original version constraints where possible.
        2. conflicting: packages that CAUSE the conflicts and cannot be reconciled
           with the rest.

        Rules:
        - Every package from the input requirements.txt must appear in exactly one group.
        - Prefer moving the fewest packages possible to "conflicting".
        - Keep original version specs exactly as given.
        - Never hallucinate package names.

        requirements.txt:
        {current_requirements}

        pip conflict errors:
        {pip_output}
        """
        ...


# ─── Manual-fix-recommendation agent (web UI, independent of auto-fix) ────────


class IssueAnalysis(BaseModel):
    """Plain-English summary and technical root cause of a pip install failure."""

    issue_summary: str
    root_cause: str


class Recommendation(BaseModel):
    """One concrete manual fix a user can follow."""

    title: str
    steps: list[str]


class ManualAnalysisAgent(Agent):
    """You are an expert Python packaging and environment engineer."""

    @strategy(PredictStrategy())
    async def analyze_issue(self, requirements: str, pip_output: str) -> IssueAnalysis:
        """Analyze a requirements.txt and the full output from attempting to install it.

        requirements.txt (original, uploaded by user):
        {requirements}

        pip install output:
        {pip_output}

        Return:
        - issue_summary: 2-4 sentence plain-English description of what was found —
          what installed successfully, what failed, and the overall state of the requirements.
        - root_cause: technical explanation of the root cause(s): version conflicts,
          overly strict pins, missing system libraries, Python version incompatibilities,
          OS dependencies, missing build tools (Rust, C compiler), deprecated packages,
          environment issues, etc. Be specific — name exact packages and versions involved.
        """
        ...

    @strategy(PredictStrategy())
    async def recommend_fixes(
        self, requirements: str, pip_output: str, root_cause: str
    ) -> list[Recommendation]:
        """Produce concrete manual fix recommendations the user can follow.

        requirements.txt (original, uploaded by user):
        {requirements}

        pip install output:
        {pip_output}

        Root cause analysis:
        {root_cause}

        Rules:
        - Always return at least one recommendation.
        - Cover every distinct issue found: version conflicts, system/OS libraries,
          build tools, Python version mismatches, network/proxy issues, deprecated
          packages, etc.
        - Use exact package names, version numbers, and runnable shell commands.
        - If install succeeded without errors, give proactive guidance: loosen overly
          strict pins, suggest pip-compile/lockfile, note deprecation risks, etc.
        """
        ...
