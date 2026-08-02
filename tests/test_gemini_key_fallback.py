"""
Rate-limited primary Gemini key falls back to the vedastro key, once.
"""

import httpx
import pytest
from unittest.mock import AsyncMock, Mock
from openai import RateLimitError

from app.services.clara_conversation_service import ClaraConversationService


def _rate_limit_error():
    return RateLimitError(
        "429 quota exceeded",
        response=httpx.Response(429, request=httpx.Request("POST", "https://x/chat")),
        body=None,
    )


def _client(side_effect=None, return_value=None):
    client = Mock()
    client.chat.completions.create = AsyncMock(side_effect=side_effect, return_value=return_value)
    return client


@pytest.mark.asyncio
async def test_429_retries_on_vedastro_key():
    service = ClaraConversationService()
    service.openai_client = _client(side_effect=_rate_limit_error())
    service.vedastro_client = _client(return_value="vedastro-reply")

    assert await service._chat_completion(model="m") == "vedastro-reply"
    service.vedastro_client.chat.completions.create.assert_awaited_once_with(model="m")


@pytest.mark.asyncio
async def test_non_rate_limit_error_is_not_retried():
    service = ClaraConversationService()
    service.openai_client = _client(side_effect=ValueError("bad request"))
    service.vedastro_client = _client(return_value="vedastro-reply")

    with pytest.raises(ValueError):
        await service._chat_completion(model="m")
    service.vedastro_client.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_429_raises_when_no_vedastro_key_configured():
    service = ClaraConversationService()
    service.openai_client = _client(side_effect=_rate_limit_error())
    service.vedastro_client = None

    with pytest.raises(RateLimitError):
        await service._chat_completion(model="m")
