"""Build the real provider adapters from environment variables.

SDK retries are switched off (max_retries=0): a retry hidden inside the SDK would inflate latency
and hide failures from the gateway, and retrying is the gateway's decision (slice 3). The timeout is
explicit for the same reason."""
import os

import anthropic
import openai

from llm_gateway.providers.anthropic_messages import AnthropicProvider
from llm_gateway.providers.base import Provider
from llm_gateway.providers.openai_compat import GROQ_BASE_URL, OpenAICompatibleProvider

TIMEOUT_SECONDS = 60.0
KEYS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "groq": "GROQ_API_KEY"}


class MissingCredentials(RuntimeError):
    pass


def build_providers(names: set[str], env=os.environ) -> dict[str, Provider]:
    """One adapter per provider name in `names`. Raises MissingCredentials naming every absent key,
    so a run fails before its first call instead of halfway through."""
    missing = sorted(KEYS[n] for n in names if not env.get(KEYS[n]))
    if missing:
        raise MissingCredentials(f"Set {', '.join(missing)} in .env (see .env.example).")
    providers: dict[str, Provider] = {}
    if "openai" in names:
        providers["openai"] = OpenAICompatibleProvider("openai", openai.AsyncOpenAI(
            api_key=env["OPENAI_API_KEY"], timeout=TIMEOUT_SECONDS, max_retries=0))
    if "groq" in names:
        providers["groq"] = OpenAICompatibleProvider("groq", openai.AsyncOpenAI(
            api_key=env["GROQ_API_KEY"], base_url=GROQ_BASE_URL, timeout=TIMEOUT_SECONDS, max_retries=0))
    if "anthropic" in names:
        providers["anthropic"] = AnthropicProvider(anthropic.AsyncAnthropic(
            api_key=env["ANTHROPIC_API_KEY"], timeout=TIMEOUT_SECONDS, max_retries=0))
    return providers
