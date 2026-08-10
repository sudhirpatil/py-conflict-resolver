"""Factory for creating NOOA LLM clients from config."""

from __future__ import annotations

import os

from nooa.unifiedllm import UnifiedLLM, get_llm_client

from conflict_resolver.config import LLMConfig

# Maps provider name → required environment variable
ENV_VARS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
}

# litellm routes OpenAI/Anthropic model names as-is; Google models need this prefix.
_MODEL_PREFIX: dict[str, str] = {"gemini": "gemini/"}


def create_llm(config: LLMConfig) -> UnifiedLLM:
    """Return a NOOA UnifiedLLM client for the configured provider and model."""
    _check_api_key(config.provider)

    if config.provider not in ENV_VARS:
        raise ValueError(
            f"Unknown provider {config.provider!r}. "
            f"Valid providers: {list(ENV_VARS)}"
        )

    model = _MODEL_PREFIX.get(config.provider, "") + config.model
    return get_llm_client(model, temperature=config.temperature)


def _check_api_key(provider: str) -> None:
    env_var = ENV_VARS.get(provider)
    if env_var and not os.environ.get(env_var):
        raise EnvironmentError(
            f"Provider {provider!r} requires the {env_var} environment variable to be set.\n"
            f"Run: export {env_var}=<your-api-key>"
        )
