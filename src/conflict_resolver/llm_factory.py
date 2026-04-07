"""Factory for creating LangChain chat model instances from config."""

from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel

from conflict_resolver.config import LLMConfig

# Maps provider name → required environment variable
ENV_VARS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
}


def create_llm(config: LLMConfig) -> BaseChatModel:
    """Return a LangChain BaseChatModel for the configured provider and model.

    Lazy-imports the provider package so only the selected provider's
    langchain package needs to be installed.
    """
    _check_api_key(config.provider)

    match config.provider:
        case "openai":
            from langchain_openai import ChatOpenAI  # type: ignore[import]

            return ChatOpenAI(model=config.model, temperature=config.temperature)

        case "anthropic":
            from langchain_anthropic import ChatAnthropic  # type: ignore[import]

            return ChatAnthropic(model=config.model, temperature=config.temperature)

        case "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore[import]

            return ChatGoogleGenerativeAI(
                model=config.model, temperature=config.temperature
            )

        case _:
            raise ValueError(
                f"Unknown provider {config.provider!r}. "
                f"Valid providers: {list(ENV_VARS)}"
            )


def _check_api_key(provider: str) -> None:
    env_var = ENV_VARS.get(provider)
    if env_var and not os.environ.get(env_var):
        raise EnvironmentError(
            f"Provider {provider!r} requires the {env_var} environment variable to be set.\n"
            f"Run: export {env_var}=<your-api-key>"
        )
