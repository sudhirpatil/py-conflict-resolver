"""LangGraph StateGraph for the conflict-resolution agent loop."""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path
from typing import Annotated, Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from conflict_resolver.prompts import (
    TOOL_USE_SYSTEM_PROMPT,
    build_tool_use_message,
    condense_pip_output,
    diff_requirements,
)
from conflict_resolver.req_tools import (
    AddPackage,
    SetPackageVersion,
    apply_tool_calls,
)
from conflict_resolver.venv_manager import InstallResult, VenvManager

logger = logging.getLogger(__name__)

# ─── State ────────────────────────────────────────────────────────────────────


class ResolverState(TypedDict):
    # Immutable inputs
    original_requirements_path: str
    original_requirements: str

    # Mutable working state
    current_requirements: str
    attempt_count: int

    # Result of the last install attempt
    last_install_success: bool
    last_pip_output: str
    last_dry_run_output: str

    # History of failed attempts passed to LLM for context
    failed_attempts: list[dict[str, str]]

    # LangChain messages (append-only via add_messages reducer)
    messages: Annotated[list, add_messages]

    # Terminal outputs
    resolved_requirements: str | None
    error_message: str | None

    # PyPI version cache — populated by pypi_lookup node, reused across iterations
    pypi_versions: dict[str, list[str]]
    # requires_dist for pinned versions — {package: [dep strings]} e.g. {"django": ["asgiref>=3.4.1"]}
    pypi_requires_dist: dict[str, list[str]]


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _packages_from_requirements(requirements_text: str) -> list[str]:
    """Return package names parsed from a requirements text block."""
    from conflict_resolver.req_tools import _REQ_LINE_RE
    names = []
    for line in requirements_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        m = _REQ_LINE_RE.match(stripped)
        if m:
            names.append(m.group(1))
    return names


def _pinned_versions_from_requirements(requirements_text: str) -> dict[str, str]:
    """Return {package: version} for lines with an exact pin (==x.y.z only).

    Only exact pins are returned because requires_dist must be fetched for a
    specific version — range constraints don't map to a single PyPI endpoint.
    """
    from conflict_resolver.req_tools import _REQ_LINE_RE
    import re as _re
    _exact = _re.compile(r"^==\s*(.+)$")
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
        em = _exact.match(spec)
        if em:
            result[pkg] = em.group(1).strip()
    return result


# ─── Node factories ───────────────────────────────────────────────────────────


def make_pypi_lookup_node():
    def pypi_lookup_node(state: ResolverState) -> dict[str, Any]:
        from conflict_resolver.pypi_client import fetch_requires_dist_bulk, fetch_versions_bulk

        packages = _packages_from_requirements(state["current_requirements"])
        existing_versions = state.get("pypi_versions") or {}
        existing_deps = state.get("pypi_requires_dist") or {}

        # ── Available versions ────────────────────────────────────────────────
        to_fetch = [p for p in packages if p not in existing_versions]
        if to_fetch:
            logger.info("Fetching PyPI version data for %d package(s)…", len(to_fetch))
            fresh = fetch_versions_bulk(to_fetch)
            logger.info("PyPI version lookup complete: %d/%d packages found", len(fresh), len(to_fetch))
            updated_versions = {**existing_versions, **fresh}
        else:
            logger.info("All packages already in PyPI versions cache — skipping")
            updated_versions = existing_versions

        # ── requires_dist for pinned packages ─────────────────────────────────
        pinned = _pinned_versions_from_requirements(state["current_requirements"])
        to_fetch_deps = {
            pkg: ver for pkg, ver in pinned.items() if pkg not in existing_deps
        }
        if to_fetch_deps:
            logger.info(
                "Fetching dependency metadata (requires_dist) for %d pinned package(s): %s",
                len(to_fetch_deps), ", ".join(f"{p}=={v}" for p, v in to_fetch_deps.items()),
            )
            fresh_deps = fetch_requires_dist_bulk(to_fetch_deps)
            logger.info(
                "requires_dist lookup complete: %d/%d packages have declared dependencies",
                len(fresh_deps), len(to_fetch_deps),
            )
            updated_deps = {**existing_deps, **fresh_deps}
        else:
            skipped = set(pinned) - set(to_fetch_deps)
            if skipped:
                logger.info(
                    "requires_dist: %d package(s) already cached — skipping fetch (%s)",
                    len(skipped), ", ".join(sorted(skipped)),
                )
            else:
                logger.info(
                    "requires_dist: no exact-pinned packages found in requirements "
                    "(only exact ==x.y.z pins are looked up; range constraints are skipped)"
                )
            updated_deps = existing_deps

        return {"pypi_versions": updated_versions, "pypi_requires_dist": updated_deps}

    return pypi_lookup_node


def make_install_node(venv_manager: VenvManager, pip_timeout: int):
    def install_node(state: ResolverState) -> dict[str, Any]:
        attempt = state["attempt_count"] + 1
        logger.info("Install attempt %d", attempt)

        # Write current requirements to a temp file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            prefix="requirements_",
            delete=False,
        ) as f:
            f.write(state["current_requirements"])
            req_file = Path(f.name)

        try:
            logger.info("Running pip dry-run to detect conflicts before full install…")
            has_conflicts, dry_run_output = venv_manager.dry_run_install(req_file)

            if has_conflicts:
                # Dry-run already confirmed conflicts — skip the expensive real install.
                # Use the dry-run output as pip output so the LLM has full conflict details.
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
                # Dry-run clean (or dry-run unavailable/timed out) — run real install
                # to actually write packages to the venv and confirm success.
                result = venv_manager.install_from_file(req_file, timeout=pip_timeout)
        finally:
            req_file.unlink(missing_ok=True)

        update: dict[str, Any] = {
            "last_install_success": result.success,
            "last_pip_output": result.combined_output,
            "last_dry_run_output": dry_run_output,
            "attempt_count": attempt,
        }
        if result.success:
            update["resolved_requirements"] = state["current_requirements"]

        return update

    return install_node


def make_analyze_node(llm: BaseChatModel):
    llm_with_tools = llm.bind_tools([SetPackageVersion, AddPackage])

    def analyze_node(state: ResolverState) -> dict[str, Any]:
        attempt = state["attempt_count"]
        logger.info("─" * 60)
        logger.info("Analyzing pip failure with LLM (attempt %d)", attempt)
        logger.info("─" * 60)

        logger.info("pip error summary:")
        # Show the last non-empty stderr lines for a quick summary at INFO level
        for line in state["last_pip_output"].splitlines():
            stripped = line.strip()
            if stripped and ("error" in stripped.lower() or "conflict" in stripped.lower()):
                logger.info("  %s", stripped)
        logger.debug("Full pip output:\n%s", state["last_pip_output"])

        if state["failed_attempts"]:
            logger.info(
                "Including %d previous failed attempt(s) in LLM context to avoid repeats",
                len(state["failed_attempts"]),
            )

        pypi_versions = state.get("pypi_versions") or {}
        pypi_requires_dist = state.get("pypi_requires_dist") or {}
        dry_run_output = state.get("last_dry_run_output") or ""

        logger.info(
            "Prompt sections: pypi_versions=%d pkg(s), requires_dist=%d pkg(s), dry_run=%s",
            len(pypi_versions),
            len(pypi_requires_dist),
            "yes" if dry_run_output else "no",
        )
        if pypi_requires_dist:
            for pkg, deps in sorted(pypi_requires_dist.items()):
                logger.info("  requires_dist %s: %s", pkg, ", ".join(deps))

        logger.info("Sending prompt to LLM (with tool use)…")
        human_msg = build_tool_use_message(
            attempt=attempt,
            max_loops=_get_max_loops_from_state(state),
            requirements_content=state["current_requirements"],
            pip_output=state["last_pip_output"],
            failed_attempts=state["failed_attempts"],
            pypi_versions=pypi_versions,
            dry_run_output=dry_run_output,
            pypi_requires_dist=pypi_requires_dist,
        )
        logger.debug("LLM prompt (human message):\n%s", human_msg)

        response = llm_with_tools.invoke(
            [SystemMessage(content=TOOL_USE_SYSTEM_PROMPT), HumanMessage(content=human_msg)]
        )

        raw_response = response.content if isinstance(response.content, str) else str(response.content)
        logger.info("LLM response received (%d chars)", len(raw_response))
        logger.debug("Raw LLM response:\n%s", raw_response)

        # Store a compact diff (not the full requirements) and condensed pip
        # errors (no download noise) to keep token usage low across iterations.
        prev_requirements = (
            state["failed_attempts"][-1]["_full_requirements"]
            if state["failed_attempts"]
            else state["original_requirements"]
        )
        new_failed = state["failed_attempts"] + [
            {
                "requirements": diff_requirements(prev_requirements, state["current_requirements"]),
                "pip_output": condense_pip_output(state["last_pip_output"]),
                "_full_requirements": state["current_requirements"],  # kept for next diff, not sent to LLM
            }
        ]

        return {
            "messages": [HumanMessage(content=human_msg), response],
            "failed_attempts": new_failed,
        }

    return analyze_node


def make_fix_node():
    def fix_node(state: ResolverState) -> dict[str, Any]:
        # Extract the last AIMessage
        last_ai_msg: AIMessage | None = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage):
                last_ai_msg = msg
                break

        tool_calls = getattr(last_ai_msg, "tool_calls", None) or []

        if tool_calls:
            logger.info("Applying %d tool call(s) from LLM…", len(tool_calls))
            fixed = apply_tool_calls(state["current_requirements"], tool_calls)
        else:
            # Fallback: parse raw text (existing logic)
            logger.info("No tool calls in LLM response — falling back to text parsing…")
            last_ai: str = (
                last_ai_msg.content
                if last_ai_msg and isinstance(last_ai_msg.content, str)
                else str(last_ai_msg.content) if last_ai_msg else ""
            )
            had_fences = bool(_FENCE_RE.search(last_ai))
            fixed = _strip_markdown_fences(last_ai).strip()
            if had_fences:
                logger.debug("Stripped markdown fences from LLM output")

        # Validate: must have at least one non-comment, non-empty line
        non_empty = [
            line for line in fixed.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if not non_empty:
            logger.warning(
                "LLM returned an empty or comment-only requirements block — "
                "retaining previous requirements"
            )
            fixed = state["current_requirements"]
        else:
            prev_lines = set(state["current_requirements"].splitlines())
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
            logger.debug("Fixed requirements:\n%s", fixed)

        logger.info("Proceeding to next install attempt…")
        logger.info("─" * 60)
        return {"current_requirements": fixed}

    return fix_node


def make_finish_node(max_loops: int):
    def finish_node(state: ResolverState) -> dict[str, Any]:
        if state["last_install_success"]:
            logger.info(
                "Resolved successfully after %d attempt(s)", state["attempt_count"]
            )
            return {}
        else:
            msg = (
                f"Could not resolve conflicts after {state['attempt_count']} attempt(s) "
                f"(max_loops={max_loops})."
            )
            logger.error(msg)
            return {"error_message": msg}

    return finish_node


# ─── Routing ──────────────────────────────────────────────────────────────────


def _get_max_loops_from_state(state: ResolverState) -> int:
    # max_loops is not stored in state; it's captured via closure in the router.
    # This helper is used only for prompt building — we pass a sentinel 0 when unknown.
    return 0


def make_router(max_loops: int):
    def route_after_install(state: ResolverState) -> str:
        if state["last_install_success"]:
            return "finish"
        if state["attempt_count"] >= max_loops:
            return "finish"
        return "analyze"

    return route_after_install


# ─── Graph builder ────────────────────────────────────────────────────────────


def build_graph(llm: BaseChatModel, venv_manager: VenvManager, max_loops: int, pip_timeout: int = 300, pypi_lookup_enabled: bool = True):
    """Build and compile the LangGraph conflict-resolution graph."""
    graph = StateGraph(ResolverState)

    graph.add_node("install", make_install_node(venv_manager, pip_timeout))
    graph.add_node("analyze", make_analyze_node(llm))
    graph.add_node("fix", make_fix_node())
    graph.add_node("finish", make_finish_node(max_loops))

    graph.set_entry_point("install")

    if pypi_lookup_enabled:
        logger.info("PyPI lookup node enabled")
        graph.add_node("pypi_lookup", make_pypi_lookup_node())
        graph.add_conditional_edges(
            "install",
            make_router(max_loops),
            {"finish": "finish", "analyze": "pypi_lookup"},
        )
        graph.add_edge("pypi_lookup", "analyze")
    else:
        logger.info("PyPI lookup node disabled — skipping directly to analyze")
        graph.add_conditional_edges(
            "install",
            make_router(max_loops),
            {"finish": "finish", "analyze": "analyze"},
        )

    graph.add_edge("analyze", "fix")
    graph.add_edge("fix", "install")
    graph.add_edge("finish", END)

    return graph.compile()


# ─── Helpers ──────────────────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"```(?:python|text|plaintext|requirements)?\s*\n?(.*?)```", re.DOTALL)


def _strip_markdown_fences(text: str) -> str:
    """Remove ```...``` fences if the LLM wrapped its output in them."""
    match = _FENCE_RE.search(text)
    if match:
        return match.group(1)
    return text
