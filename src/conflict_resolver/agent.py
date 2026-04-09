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
    SYSTEM_PROMPT,
    build_user_message,
    condense_pip_output,
    diff_requirements,
)
from conflict_resolver.venv_manager import VenvManager

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

    # History of failed attempts passed to LLM for context
    failed_attempts: list[dict[str, str]]

    # LangChain messages (append-only via add_messages reducer)
    messages: Annotated[list, add_messages]

    # Terminal outputs
    resolved_requirements: str | None
    error_message: str | None


# ─── Node factories ───────────────────────────────────────────────────────────


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
            result = venv_manager.install_from_file(req_file, timeout=pip_timeout)
        finally:
            req_file.unlink(missing_ok=True)

        update: dict[str, Any] = {
            "last_install_success": result.success,
            "last_pip_output": result.combined_output,
            "attempt_count": attempt,
        }
        if result.success:
            update["resolved_requirements"] = state["current_requirements"]

        return update

    return install_node


def make_analyze_node(llm: BaseChatModel):
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

        logger.info("Sending prompt to LLM…")
        human_msg = build_user_message(
            attempt=attempt,
            max_loops=_get_max_loops_from_state(state),
            requirements_content=state["current_requirements"],
            pip_output=state["last_pip_output"],
            failed_attempts=state["failed_attempts"],
        )
        logger.debug("LLM prompt (human message):\n%s", human_msg)

        response = llm.invoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=human_msg)]
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
        # Extract the last AIMessage content
        last_ai: str = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage):
                last_ai = msg.content if isinstance(msg.content, str) else str(msg.content)
                break

        logger.info("Parsing LLM response into requirements…")
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


def build_graph(llm: BaseChatModel, venv_manager: VenvManager, max_loops: int, pip_timeout: int = 300):
    """Build and compile the LangGraph conflict-resolution graph."""
    graph = StateGraph(ResolverState)

    graph.add_node("install", make_install_node(venv_manager, pip_timeout))
    graph.add_node("analyze", make_analyze_node(llm))
    graph.add_node("fix", make_fix_node())
    graph.add_node("finish", make_finish_node(max_loops))

    graph.set_entry_point("install")

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
