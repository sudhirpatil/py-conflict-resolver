"""Lightweight PyPI JSON API client — no third-party HTTP dependency."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

_PYPI_TIMEOUT = 5  # seconds per request


def fetch_versions(package: str) -> list[str]:
    """Return available versions for *package* from PyPI, newest-first.

    Returns [] on any error (network, 404, etc.) — caller must handle gracefully.
    """
    url = f"https://pypi.org/pypi/{package}/json"
    try:
        with urllib.request.urlopen(url, timeout=_PYPI_TIMEOUT) as resp:
            data = json.loads(resp.read())
        releases = data.get("releases", {})
        # Filter out yanked-only releases and empty upload lists
        versions = [
            v for v, files in releases.items()
            if files and not all(f.get("yanked") for f in files)
        ]
        # Sort descending by parsed version
        from packaging.version import InvalidVersion, Version
        valid: list[tuple[Version, str]] = []
        for v in versions:
            try:
                valid.append((Version(v), v))
            except InvalidVersion:
                pass
        valid.sort(reverse=True)
        return [v_str for _, v_str in valid]
    except Exception as exc:
        logger.debug("PyPI lookup failed for %r: %s", package, exc)
        return []


def fetch_versions_bulk(packages: list[str], max_workers: int = 8) -> dict[str, list[str]]:
    """Fetch PyPI versions for multiple packages concurrently.

    Returns {package: [versions]} — packages that fail are absent from the result.
    """
    result: dict[str, list[str]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_versions, pkg): pkg for pkg in packages}
        for future in as_completed(futures):
            pkg = futures[future]
            versions = future.result()
            if versions:
                result[pkg] = versions
                logger.debug(
                    "PyPI: %s — %d versions found (latest: %s)", pkg, len(versions), versions[0]
                )
            else:
                logger.debug("PyPI: %s — not found or lookup failed", pkg)
    return result


def fetch_requires_dist(package: str, version: str) -> list[str]:
    """Return the requires_dist list for a specific package version from PyPI.

    Queries https://pypi.org/pypi/<package>/<version>/json and returns
    info.requires_dist (list of PEP 508 dependency strings).
    Returns [] on any error.
    """
    url = f"https://pypi.org/pypi/{package}/{version}/json"
    try:
        with urllib.request.urlopen(url, timeout=_PYPI_TIMEOUT) as resp:
            data = json.loads(resp.read())
        deps = data.get("info", {}).get("requires_dist") or []
        logger.debug("PyPI requires_dist: %s==%s — %d deps", package, version, len(deps))
        return deps
    except Exception as exc:
        logger.debug("PyPI requires_dist failed for %r==%r: %s", package, version, exc)
        return []


def fetch_requires_dist_bulk(
    package_versions: dict[str, str], max_workers: int = 8
) -> dict[str, list[str]]:
    """Fetch requires_dist for multiple package==version pairs concurrently.

    Args:
        package_versions: {package_name: pinned_version} e.g. {"django": "3.2"}

    Returns {package: [requires_dist strings]} — failures are absent from result.
    """
    result: dict[str, list[str]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(fetch_requires_dist, pkg, ver): pkg
            for pkg, ver in package_versions.items()
        }
        for future in as_completed(futures):
            pkg = futures[future]
            deps = future.result()
            if deps:
                result[pkg] = deps
    return result
