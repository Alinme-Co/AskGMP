"""
openai_client.py — one place that constructs an OpenAI client.

Four call sites need one (the indexer, the retriever, the chat service, the
upload index) and each used to carry its own "if not OPENAI_API_KEY: raise"
block. The failure they guard against is worth one clear message rather than
four slightly different ones — a missing key surfaces here, at construction,
instead of as an authentication error deep inside a batch of embeddings.

Import is cheap: the openai package is imported inside the function, so a
--dry-run or a unit test that never embeds anything does not pay for it.
"""

from __future__ import annotations

from app import config


def get_openai_client(api_key: str | None = None):
    """An OpenAI client, or a clear error about the missing key."""
    from openai import OpenAI

    if not api_key and config.EMBED_VIA_OPENROUTER:
        return get_chat_client()

    key = api_key or config.OPENAI_API_KEY
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set — check .env")
    return OpenAI(api_key=key)


def get_chat_client(api_key: str | None = None):
    """The chat client: OpenRouter through the OpenAI SDK."""
    from openai import OpenAI

    key = api_key or config.OPENROUTER_API_KEY
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set — check .env")
    return OpenAI(base_url=config.LLM_BASE_URL, api_key=key)
