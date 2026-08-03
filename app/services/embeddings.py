"""
Text embeddings for semantic recall over conversation_log.

Gemini's OpenAI-compatible endpoint, same two-key 429 fallback as
ClaraConversationService._chat_completion. Every caller here is best-effort:
embed_texts returns None rather than raising, because a missing embedding only
costs recall quality, never a conversation turn.
"""

import logging
from typing import List, Optional, Sequence

from openai import AsyncOpenAI, RateLimitError

from app.core.config import settings

logger = logging.getLogger(__name__)

EMBED_MODEL = "gemini-embedding-001"
# 768 over the 3072 default: the recall win past 768 is noise, and the column,
# the index and every row get 4x smaller.
EMBED_DIM = 768
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Free tier is 100 embedding req/min per key. One request carries a whole batch,
# so this is a per-batch budget, not a per-text one.
EMBED_RPM = 100

_clients: Optional[List[AsyncOpenAI]] = None


def _get_clients() -> List[AsyncOpenAI]:
    """Primary key first, vedastro key as the 429 understudy. Built once."""
    global _clients
    if _clients is None:
        _clients = [
            AsyncOpenAI(api_key=key, base_url=GEMINI_BASE_URL)
            for key in (settings.gemini_api_key, settings.vedastro_gemini_api_key)
            if key
        ]
    return _clients


async def embed_texts(texts: Sequence[str]) -> Optional[List[List[float]]]:
    """Embed a batch in one request. None on any failure — callers degrade, never fail.

    Order and length of the result match `texts`.
    """
    texts = [t for t in texts]
    if not texts:
        return []

    clients = _get_clients()
    if not clients:
        logger.debug("No Gemini key configured, skipping embeddings")
        return None

    for i, client in enumerate(clients):
        try:
            resp = await client.embeddings.create(
                model=EMBED_MODEL, input=list(texts), dimensions=EMBED_DIM
            )
            # Gemini's compat endpoint leaves `index` unset and returns input order.
            # Sort only when it actually tells us the order.
            data = list(resp.data)
            if all(getattr(d, "index", None) is not None for d in data):
                data.sort(key=lambda d: d.index)
            return [d.embedding for d in data]
        except Exception as e:
            rate_limited = isinstance(e, RateLimitError) or getattr(e, "status_code", None) == 429
            if rate_limited and i + 1 < len(clients):
                logger.warning("Embedding key %d rate-limited, trying the next one", i)
                continue
            logger.warning("Embedding call failed (%s), caller falls back", e)
            return None
    return None


async def embed_one(text: str) -> Optional[List[float]]:
    vectors = await embed_texts([text])
    return vectors[0] if vectors else None


def to_pgvector(vector: Sequence[float]) -> str:
    """pgvector's text input form. Bind as a string and CAST(:v AS vector) in SQL.

    ponytail: 3 lines instead of the pgvector python package — nothing else in
    the codebase needs vector types in the ORM layer.
    """
    return "[" + ",".join(f"{x:.7g}" for x in vector) + "]"
