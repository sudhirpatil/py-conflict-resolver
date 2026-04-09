"""Prompt templates for the conflict-resolver LLM agent."""

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


SYSTEM_PROMPT = """\
You are an expert Python packaging engineer specializing in resolving pip dependency conflicts.

You will be given:
1. The current contents of requirements.txt
2. The pip install output (stdout + stderr) that shows the error or conflict

Your task is to produce a FIXED requirements.txt that will install successfully.

Rules:
- Output ONLY the raw contents of the fixed requirements.txt — nothing else.
- Do NOT include markdown code fences, commentary, or explanations.
- Preserve packages that are not involved in the conflict unchanged.
- Pin versions precisely (e.g. package==1.2.3) when resolving conflicts.
- If a package must be removed entirely to resolve the conflict, omit it and add a
  comment line starting with "# REMOVED: <package> - <brief reason>".
- Prefer relaxing version constraints over removing packages when possible.
- Never hallucinate package names that do not exist on PyPI.
- If the error is due to a missing system dependency or non-resolvable at the
  requirements level, explain it in a comment at the top with "# NOTE: ..."
  and output the best possible requirements.txt you can.
"""


def build_user_message(
    attempt: int,
    max_loops: int,
    requirements_content: str,
    pip_output: str,
    failed_attempts: list[dict[str, str]],
) -> str:
    """Build the human message for the LLM on each iteration.

    failed_attempts entries are expected to already contain condensed text
    (requirements diff and condensed pip errors) to stay within token limits.
    """
    parts: list[str] = []

    parts.append(f"Attempt {attempt} of {max_loops}.")

    if failed_attempts:
        parts.append(
            f"\nPrevious failed attempts ({len(failed_attempts)}) — do NOT repeat these fixes:\n"
        )
        for i, fa in enumerate(failed_attempts, 1):
            parts.append(f"--- Attempt {i} changes (diff) ---\n{fa['requirements']}")
            parts.append(f"--- Attempt {i} pip errors ---\n{fa['pip_output']}\n")

    parts.append("--- Current requirements.txt ---")
    parts.append(requirements_content.strip())

    parts.append("\n--- pip errors ---")
    parts.append(condense_pip_output(pip_output).strip())

    parts.append(
        "\nPlease output the fixed requirements.txt now (raw text only, no markdown fences):"
    )

    return "\n".join(parts)
