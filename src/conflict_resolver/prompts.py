"""Deterministic string-processing helpers shared by the agent's generation methods.

System prompts and per-call framing now live as docstrings/parameters on
ConflictResolverAgent's generation methods (see agent.py) — NOOA's docstring
templating replaces the manual prompt-string builders this module used to hold.
"""

from __future__ import annotations

# pip output lines that contain no useful signal for conflict resolution
_PIP_NOISE_PREFIXES = (
    "collecting ",
    "downloading ",
    "using cached ",
    "installing collected",
    "successfully installed",
    "obtaining ",
    "building ",
    "running ",
    "created temporary",
    "added ",
    "preparing metadata",
    "checking ",
)

# Keywords that mark a line as meaningful for conflict resolution
_PIP_SIGNAL_KEYWORDS = (
    "error",
    "conflict",
    "cannot install",
    "incompatible",
    "resolutionimpossible",
    "could not find",
    "no matching distribution",
    "requires",
    "is not satisfied",
    "depends on",
    "warning",
)


def condense_pip_output(pip_output: str) -> str:
    """Strip download/install noise from pip output, keeping only conflict-relevant lines."""
    kept: list[str] = []
    for line in pip_output.splitlines():
        low = line.strip().lower()
        if not low:
            continue
        if any(low.startswith(p) for p in _PIP_NOISE_PREFIXES):
            continue
        if any(k in low for k in _PIP_SIGNAL_KEYWORDS):
            kept.append(line.rstrip())
        # Always keep section headers (===)
        elif low.startswith("==="):
            kept.append(line.rstrip())

    return "\n".join(kept) if kept else pip_output  # fallback: return original if nothing kept


def condense_requirements(requirements: str) -> str:
    """Strip blank lines and comments from a requirements block for compact history."""
    lines = [
        line.rstrip()
        for line in requirements.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return "\n".join(lines)


def diff_requirements(previous: str, current: str) -> str:
    """Return a compact +/- diff of two requirements strings."""
    prev_lines = set(condense_requirements(previous).splitlines())
    curr_lines = set(condense_requirements(current).splitlines())
    removed = sorted(prev_lines - curr_lines)
    added = sorted(curr_lines - prev_lines)
    parts = [f"- {l}" for l in removed] + [f"+ {l}" for l in added]
    return "\n".join(parts) if parts else "(no changes)"
