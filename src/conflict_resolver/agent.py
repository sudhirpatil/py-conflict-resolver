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

from conflict_resolver.prompts import SYSTEM_PROMPT, build_user_message
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


def make_install_node(venv_manager: VenvManager):
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
            result = venv_manager.install_from_file(req_file)
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
        logger.info("Analyzing pip failure with LLM (attempt %d)", state["attempt_count"])

        human_msg = build_user_message(
            attempt=state["attempt_count"],
            max_loops=_get_max_loops_from_state(state),
            requirements_content=state["current_requirements"],
            pip_output=state["last_pip_output"],
            failed_attempts=state["failed_attempts"],
        )

        response = llm.invoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=human_msg)]
        )

        new_failed = state["failed_attempts"] + [
            {
                "requirements": state["current_requirements"],
                "pip_output": state["last_pip_output"],
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

        fixed = _strip_markdown_fences(last_ai).strip()

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
            logger.debug("LLM proposed fixed requirements:\n%s", fixed)

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


def build_graph(llm: BaseChatModel, venv_manager: VenvManager, max_loops: int):
    """Build and compile the LangGraph conflict-resolution graph."""
    graph = StateGraph(ResolverState)

    graph.add_node("install", make_install_node(venv_manager))
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
