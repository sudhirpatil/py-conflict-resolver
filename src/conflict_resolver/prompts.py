"""Prompt templates for the conflict-resolver LLM agent."""

from __future__ import annotations

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
    """Build the human message for the LLM on each iteration."""
    parts: list[str] = []

    parts.append(f"Attempt {attempt} of {max_loops}.")

    if failed_attempts:
        parts.append(
            f"\nPrevious failed attempts ({len(failed_attempts)}) — do NOT repeat these fixes:\n"
        )
        for i, fa in enumerate(failed_attempts, 1):
            parts.append(f"--- Failed attempt {i} requirements.txt ---\n{fa['requirements']}")
            parts.append(f"--- Failed attempt {i} pip error ---\n{fa['pip_output']}\n")

    parts.append("--- Current requirements.txt ---")
    parts.append(requirements_content.strip())

    parts.append("\n--- pip install output ---")
    parts.append(pip_output.strip())

    parts.append(
        "\nPlease output the fixed requirements.txt now (raw text only, no markdown fences):"
    )

    return "\n".join(parts)
