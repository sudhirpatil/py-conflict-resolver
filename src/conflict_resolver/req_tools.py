"""Structured fix-action schemas and application logic for LLM-driven requirements editing."""

from __future__ import annotations

import logging
import re
from typing import Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ─── Fix-action schemas (used as NOOA generation-method return types) ────────


class SetPackageVersion(BaseModel):
    """Set or update the version constraint for a package that is causing a conflict."""

    kind: Literal["set_package_version"] = "set_package_version"
    package: str = Field(description="Package name (e.g. 'numpy')")
    version_spec: str = Field(
        description="New version specifier (e.g. '==1.24.4' or '>=1.0,<2.0')"
    )


class RemovePackage(BaseModel):
    """Remove a package from requirements.txt when no compatible version exists."""

    kind: Literal["remove_package"] = "remove_package"
    package: str = Field(description="Package name to remove")
    reason: str = Field(description="Brief reason why this package must be removed")


class AddPackage(BaseModel):
    """Add a new package pin (e.g. to pin a transitive dependency that causes the conflict)."""

    kind: Literal["add_package"] = "add_package"
    package: str = Field(description="Package name to add")
    version_spec: str = Field(description="Version specifier (e.g. '==1.2.3')")


FixAction = SetPackageVersion | RemovePackage | AddPackage

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


def apply_fix_actions(requirements_text: str, actions: list[FixAction]) -> str:
    """Apply a list of LLM-proposed fix actions to *requirements_text* and return the result.

    Actions are applied in order; later actions see the result of earlier ones.
    """
    lines = requirements_text.splitlines()

    for action in actions:
        if isinstance(action, SetPackageVersion):
            logger.info("Fix action: set_package_version %s %s", action.package, action.version_spec)
            idx = _find_line_index(lines, action.package)
            if idx is not None:
                # Preserve any trailing comment on the original line
                m = _REQ_LINE_RE.match(lines[idx].strip())
                comment = m.group(3) or "" if m else ""
                lines[idx] = f"{action.package}{action.version_spec}{comment}"
            else:
                logger.info(
                    "Package %r not found in requirements — appending new line", action.package
                )
                lines.append(f"{action.package}{action.version_spec}")

        elif isinstance(action, RemovePackage):
            logger.info("Fix action: remove_package %s (%s)", action.package, action.reason)
            idx = _find_line_index(lines, action.package)
            if idx is not None:
                lines[idx] = f"# REMOVED: {action.package} - {action.reason}"
            else:
                logger.warning(
                    "remove_package: package %r not found in requirements", action.package
                )

        elif isinstance(action, AddPackage):
            logger.info("Fix action: add_package %s%s", action.package, action.version_spec)
            # Only add if not already present
            if _find_line_index(lines, action.package) is None:
                lines.append(f"{action.package}{action.version_spec}")
            else:
                logger.info(
                    "add_package: %r already present — use set_package_version to update it",
                    action.package,
                )

        else:
            logger.warning("Unknown fix action type %r — skipping", type(action).__name__)

    return "\n".join(lines)
