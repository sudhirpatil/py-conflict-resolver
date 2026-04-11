"""Structured tool schemas and application logic for LLM-driven requirements editing."""

from __future__ import annotations

import logging
import re

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ─── Tool schemas (passed to llm.bind_tools()) ───────────────────────────────


class SetPackageVersion(BaseModel):
    """Set or update the version constraint for a package that is causing a conflict."""

    package: str = Field(description="Package name (e.g. 'numpy')")
    version_spec: str = Field(
        description="New version specifier (e.g. '==1.24.4' or '>=1.0,<2.0')"
    )


class RemovePackage(BaseModel):
    """Remove a package from requirements.txt when no compatible version exists."""

    package: str = Field(description="Package name to remove")
    reason: str = Field(description="Brief reason why this package must be removed")


class AddPackage(BaseModel):
    """Add a new package pin (e.g. to pin a transitive dependency that causes the conflict)."""

    package: str = Field(description="Package name to add")
    version_spec: str = Field(description="Version specifier (e.g. '==1.2.3')")


# ─── Line matching helpers ────────────────────────────────────────────────────

# Matches a requirements line: package name + optional version spec + optional comment
_REQ_LINE_RE = re.compile(
    r"^([A-Za-z0-9][A-Za-z0-9._-]*)(\s*(?:[><=!~][^\s#]*)?)(\s*#.*)?$"
)


def _normalize(name: str) -> str:
    """Normalize a package name for comparison (PEP 503 canonical form)."""
    return name.lower().replace("-", "_").replace(".", "_")


def _find_line_index(lines: list[str], package: str) -> int | None:
    """Return the index of the line for *package*, or None if not found."""
    target = _normalize(package)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        m = _REQ_LINE_RE.match(stripped)
        if m and _normalize(m.group(1)) == target:
            return i
    return None


# ─── Public API ───────────────────────────────────────────────────────────────


def apply_tool_calls(requirements_text: str, tool_calls: list[dict]) -> str:
    """Apply a list of LLM tool calls to *requirements_text* and return the result.

    Each element of *tool_calls* is a dict with at minimum:
      - "name": one of "SetPackageVersion", "RemovePackage", "AddPackage"
      - "args": dict of keyword arguments matching the corresponding schema

    Tool calls are applied in order; later calls see the result of earlier ones.
    Unknown tool names are logged and skipped.
    """
    lines = requirements_text.splitlines()

    for call in tool_calls:
        name = call.get("name", "")
        args = call.get("args", {})

        if name == "SetPackageVersion":
            package = args.get("package", "")
            version_spec = args.get("version_spec", "")
            logger.info("Tool call: set_package_version %s %s", package, version_spec)
            idx = _find_line_index(lines, package)
            if idx is not None:
                # Preserve any trailing comment on the original line
                m = _REQ_LINE_RE.match(lines[idx].strip())
                comment = m.group(3) or "" if m else ""
                lines[idx] = f"{package}{version_spec}{comment}"
            else:
                logger.info(
                    "Package %r not found in requirements — appending new line", package
                )
                lines.append(f"{package}{version_spec}")

        elif name == "RemovePackage":
            package = args.get("package", "")
            reason = args.get("reason", "")
            logger.info("Tool call: remove_package %s (%s)", package, reason)
            idx = _find_line_index(lines, package)
            if idx is not None:
                lines[idx] = f"# REMOVED: {package} - {reason}"
            else:
                logger.warning(
                    "remove_package: package %r not found in requirements", package
                )

        elif name == "AddPackage":
            package = args.get("package", "")
            version_spec = args.get("version_spec", "")
            logger.info("Tool call: add_package %s%s", package, version_spec)
            # Only add if not already present
            if _find_line_index(lines, package) is None:
                lines.append(f"{package}{version_spec}")
            else:
                logger.info(
                    "add_package: %r already present — use set_package_version to update it",
                    package,
                )

        else:
            logger.warning("Unknown tool call name %r — skipping", name)

    return "\n".join(lines)
