"""Load and validate configuration from config.toml."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path


@dataclass
class LLMConfig:
    provider: str
    model: str
    temperature: float


@dataclass
class AgentConfig:
    max_loops: int
    pip_timeout: int
    pypi_lookup_enabled: bool = True


@dataclass
class AppConfig:
    llm: LLMConfig
    agent: AgentConfig
    available_models: dict[str, list[str]] = field(default_factory=dict)


def _parse_toml(data: dict) -> AppConfig:
    llm_data = data.get("llm", {})
    agent_data = data.get("agent", {})
    models_data = data.get("models", {})

    llm = LLMConfig(
        provider=llm_data.get("provider", "openai"),
        model=llm_data.get("model", "gpt-4.1"),
        temperature=float(llm_data.get("temperature", 0.2)),
    )
    agent = AgentConfig(
        max_loops=int(agent_data.get("max_loops", 10)),
        pip_timeout=int(agent_data.get("pip_timeout", 300)),
        pypi_lookup_enabled=bool(agent_data.get("pypi_lookup_enabled", True)),
    )
    available_models = {
        provider: cfg.get("available", [])
        for provider, cfg in models_data.items()
    }
    return AppConfig(llm=llm, agent=agent, available_models=available_models)


def _validate(config: AppConfig) -> None:
    allowed = config.available_models.get(config.llm.provider)
    if allowed is None:
        raise ValueError(
            f"Unknown provider {config.llm.provider!r}. "
            f"Valid providers: {list(config.available_models)}"
        )
    if allowed and config.llm.model not in allowed:
        raise ValueError(
            f"Model {config.llm.model!r} is not in the allowed list for provider "
            f"{config.llm.provider!r}.\nAllowed models: {allowed}"
        )


def _bundled_config_path() -> Path:
    """Return path to the config.toml bundled with the package (project root)."""
    # Walk up from this file to find config.toml at project root
    here = Path(__file__).parent
    for _ in range(5):
        candidate = here / "config.toml"
        if candidate.exists():
            return candidate
        here = here.parent
    raise FileNotFoundError("Bundled config.toml not found")


def load_config(path: Path | None = None) -> AppConfig:
    """Load config with search order: explicit path → CWD/config.toml → bundled default."""
    if path is not None:
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
    else:
        cwd_config = Path.cwd() / "config.toml"
        if cwd_config.exists():
            config_path = cwd_config
        else:
            config_path = _bundled_config_path()

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    config = _parse_toml(data)
    _validate(config)
    return config
