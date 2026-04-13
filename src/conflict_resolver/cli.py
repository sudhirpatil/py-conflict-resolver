"""Command-line interface for py-conflict-resolver."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from conflict_resolver.agent import ResolverState, build_graph
from conflict_resolver.config import load_config
from conflict_resolver.llm_factory import create_llm
from conflict_resolver.venv_manager import VenvManager


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="conflict-resolver",
        description=(
            "Resolve Python package version conflicts in a requirements.txt file "
            "using an LLM-powered agent loop."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  conflict-resolver requirements.txt
  conflict-resolver requirements.txt --output resolved.txt --verbose
  conflict-resolver requirements.txt --provider anthropic --model claude-sonnet-4-6
  conflict-resolver requirements.txt --max-loops 5 --config ./my-config.toml
  conflict-resolver requirements.txt --env-file /path/to/.env
  conflict-resolver requirements.txt --python 3.11
  conflict-resolver requirements.txt --python python3.12
  conflict-resolver requirements.txt --python /usr/local/bin/python3.10

Environment variables (set directly or via .env file):
  OPENAI_API_KEY     Required when provider is "openai"
  ANTHROPIC_API_KEY  Required when provider is "anthropic"
  GOOGLE_API_KEY     Required when provider is "gemini"

.env file:
  Copy .env.example to .env in your working directory and fill in your API key.
  A .env in the current directory is loaded automatically if --env-file is not given.
""",
    )
    parser.add_argument(
        "requirements",
        type=Path,
        metavar="REQUIREMENTS_FILE",
        help="Path to the input requirements.txt file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        metavar="OUTPUT_FILE",
        help=(
            "Output path for the resolved requirements.txt "
            "(default: <input>.resolved.txt)"
        ),
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=None,
        metavar="CONFIG_FILE",
        help="Path to config.toml (default: CWD/config.toml or bundled default)",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "gemini"],
        default=None,
        help="Override LLM provider from config",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="MODEL_NAME",
        help="Override LLM model from config",
    )
    parser.add_argument(
        "--max-loops",
        type=int,
        default=None,
        metavar="N",
        help="Override max agent iterations from config (default: 10)",
    )
    parser.add_argument(
        "--python",
        default=None,
        metavar="PYTHON",
        help=(
            "Python version or executable for the temp venv "
            "(e.g. '3.11', 'python3.12', '/usr/local/bin/python3.10'). "
            "Defaults to the current interpreter."
        ),
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        metavar="ENV_FILE",
        help="Path to a .env file to load (default: .env in CWD if it exists)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Load .env file before any env var reads (API keys, config overrides)
    if args.env_file is not None:
        env_path = args.env_file
        if not env_path.exists():
            # Can't use logger yet — basicConfig not called
            print(f"ERROR: .env file not found: {env_path}", file=sys.stderr)
            sys.exit(1)
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        # Auto-load .env from CWD if present (does nothing if file absent)
        load_dotenv(override=False)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=log_level,
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # Validate input file
    requirements_path: Path = args.requirements.resolve()
    if not requirements_path.exists():
        logger.error("Requirements file not found: %s", requirements_path)
        sys.exit(1)
    if not requirements_path.is_file():
        logger.error("Not a file: %s", requirements_path)
        sys.exit(1)

    # Load and override config
    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Configuration error: %s", e)
        sys.exit(1)

    if args.provider:
        config.llm.provider = args.provider
    if args.model:
        config.llm.model = args.model
    if args.max_loops is not None:
        config.agent.max_loops = args.max_loops

    # Determine output path
    output_path: Path = args.output or requirements_path.with_suffix(".resolved.txt")

    # Read input
    requirements_text = requirements_path.read_text(encoding="utf-8")

    logger.info("Input:    %s", requirements_path)
    logger.info("Output:   %s", output_path)
    logger.info("Provider: %s / %s", config.llm.provider, config.llm.model)
    logger.info("Max loops: %d", config.agent.max_loops)
    logger.info("Python:   %s", args.python or f"current interpreter ({sys.version.split()[0]})")

    # Build LLM
    try:
        llm = create_llm(config.llm)
    except (EnvironmentError, ValueError) as e:
        logger.error("%s", e)
        sys.exit(1)

    # Run agent with venv lifecycle managed by context manager
    final_state: ResolverState | None = None
    try:
        with VenvManager(python=args.python) as vm:
            graph = build_graph(llm, vm, config.agent.max_loops, config.agent.pip_timeout)

            initial_state: ResolverState = {
                "original_requirements_path": str(requirements_path),
                "original_requirements": requirements_text,
                "current_requirements": requirements_text,
                "attempt_count": 0,
                "last_install_success": False,
                "last_pip_output": "",
                "last_dry_run_output": "",
                "failed_attempts": [],
                "messages": [],
                "resolved_requirements": None,
                "error_message": None,
                "pypi_versions": {},
            }

            logger.info("Starting conflict-resolution agent loop…")
            final_state = graph.invoke(initial_state)

    except KeyboardInterrupt:
        logger.error("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=args.verbose)
        sys.exit(1)

    # Evaluate outcome
    if final_state and final_state.get("last_install_success") and final_state.get("resolved_requirements"):
        output_path.write_text(final_state["resolved_requirements"], encoding="utf-8")
        logger.info("Done! Resolved requirements written to: %s", output_path)
        sys.exit(0)
    else:
        error = (
            final_state.get("error_message") if final_state else "agent did not complete"
        )
        logger.error("Failed to resolve conflicts: %s", error)
        attempts = final_state.get("attempt_count", 0) if final_state else 0
        logger.error("Gave up after %d attempt(s)", attempts)
        sys.exit(1)


if __name__ == "__main__":
    main()
