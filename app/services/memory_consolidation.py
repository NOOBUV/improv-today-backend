"""
Nightly memory consolidation - "Clara sleeps on it".

Once a day, each user's conversation of the previous day is distilled by one LLM
call into a handful of durable, third-person statements about the user and the
relationship. Those statements are stored as extra conversation_log rows with
role='memory' and conversation_id='consolidated:<date>', embedded like any other
row, so semantic recall picks them up with no retrieval changes beyond the small
MEMORY_SIMILARITY_BOOST that stops generic third-person prose being outranked by
the verbatim lines it was distilled from.

ponytail: no new table. A memory IS a line the pair "said" to each other; the
only thing conversation_log lacked was a role for it.
"""

import asyncio
import json
import logging
import re
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import distinct, select, text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from celery_app import celery_app
from app.models.conversation_log import ConversationLog

logger = logging.getLogger(__name__)

MEMORY_ROLE = "memory"
# Non-null and shaped so it can never collide with a real session id, which keeps
# the `conversation_id != current` recall filter true for every memory row.
MEMORY_CONVERSATION_PREFIX = "consolidated:"

MAX_MEMORY_CHARS = 200
MAX_MEMORIES = 8
MIN_TURNS = 4  # a two-line exchange has nothing to sleep on
MAX_TRANSCRIPT_TURNS = 400
# Gemini free tier is 15 chat req/min; one call per user, so a small gap is plenty.
PACING_SECONDS = 4.0
# Above this cosine similarity two memory statements are the same fact in
# different words. Deliberately high: re-learning a fact is cheap noise, dropping
# a genuinely new one is lost memory.
DUPLICATE_SIMILARITY = 0.92

_PROMPT = """You are the memory of a companion named Clara, consolidating one day of conversation with a user.

Distill the transcript below into 3-8 short statements worth remembering months from now: facts about the user, ongoing situations, people in their life, and emotional beats that mattered. Skip greetings, small talk, and anything about Clara's own day.

Rules:
- Write in the third person about the user ("He is...", "Her manager...").
- Each statement stands alone - no "this", "that", or "the meeting" without saying which.
- One fact per statement, at most {max_chars} characters.
- Only what the transcript supports. Invent nothing.

Return ONLY a JSON array of strings, nothing else.

TRANSCRIPT ({day}):
{transcript}"""


def memory_conversation_id(target_date: date) -> str:
    return f"{MEMORY_CONVERSATION_PREFIX}{target_date.isoformat()}"


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _is_near_duplicate(candidate: str, existing: List[str]) -> bool:
    """Normalized-exact or heavy word overlap against memories we already hold.

    The embedding-free half of dedupe: it also catches duplicates inside the same
    batch, and it is the only check that runs when embeddings are unavailable.
    """
    norm = _normalize(candidate)
    if not norm:
        return True
    words = set(norm.split())
    for other in existing:
        other_norm = _normalize(other)
        if norm == other_norm:
            return True
        other_words = set(other_norm.split())
        union = words | other_words
        if union and len(words & other_words) / len(union) >= 0.8:
            return True
    return False


async def _has_similar_memory(
    session: AsyncSession, user_id: str, vector: List[float]
) -> bool:
    """Cosine check against this user's existing memory rows.

    ponytail: one small query per candidate (3-8 a night) beats pulling every
    stored vector into python to compare it there.
    """
    try:
        from app.services.embeddings import to_pgvector

        return (await session.execute(sql_text("""
            SELECT 1 FROM conversation_log
            WHERE user_id = :user_id AND role = :role AND embedding IS NOT NULL
              AND 1 - (embedding <=> CAST(:v AS vector)) > :threshold
            LIMIT 1
        """), {
            "user_id": user_id, "role": MEMORY_ROLE,
            "v": to_pgvector(vector), "threshold": DUPLICATE_SIMILARITY,
        })).scalar_one_or_none() is not None
    except Exception as e:
        # No pgvector / no embeddings — _is_near_duplicate already ran.
        logger.warning("Memory similarity check unavailable: %s", e)
        return False


def parse_memory_statements(raw: Optional[str]) -> List[str]:
    """Pull statements out of the model's reply.

    Defensive on purpose: the model sometimes ignores the JSON envelope and just
    writes a bulleted list (same reason _parse_llm_response exists).
    """
    if not raw:
        return []

    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text).rstrip("`").strip()

    statements: List[str] = []
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start:end + 1])
            if isinstance(parsed, list):
                statements = [s for s in parsed if isinstance(s, str)]
        except json.JSONDecodeError:
            logger.warning("Memory consolidation: JSON array unparseable, falling back to lines")

    if not statements:
        statements = [re.sub(r'^[\-\*\d\.\)\s"]+|["\,]+$', "", line).strip()
                      for line in text.splitlines()]

    cleaned = []
    for s in statements:
        s = s.strip()
        if len(s) < 10:
            continue
        cleaned.append(s[:MAX_MEMORY_CHARS].rstrip())
    return cleaned[:MAX_MEMORIES]


def _day_bounds(target_date: date):
    start = datetime.combine(target_date, time.min, tzinfo=timezone.utc)
    return start, start + timedelta(days=1)


async def _already_consolidated(session: AsyncSession, user_id: str, target_date: date) -> bool:
    return (await session.execute(
        select(ConversationLog.id).where(
            ConversationLog.user_id == user_id,
            ConversationLog.conversation_id == memory_conversation_id(target_date),
        ).limit(1)
    )).scalar_one_or_none() is not None


async def consolidate_user_day(
    session: AsyncSession,
    user_id: str,
    target_date: date,
    chat_completion=None,
) -> Dict[str, Any]:
    """Distil one user's day into memory rows. Idempotent: a second run skips."""
    if await _already_consolidated(session, user_id, target_date):
        return {"user_id": user_id, "skipped": "already_consolidated", "memories": []}

    start, end = _day_bounds(target_date)
    rows = (await session.execute(
        select(ConversationLog)
        .where(
            ConversationLog.user_id == user_id,
            ConversationLog.role.in_(("user", "assistant")),
            ConversationLog.created_at >= start,
            ConversationLog.created_at < end,
        )
        .order_by(ConversationLog.created_at, ConversationLog.id)
        .limit(MAX_TRANSCRIPT_TURNS)
    )).scalars().all()

    if len(rows) < MIN_TURNS:
        return {"user_id": user_id, "skipped": "too_little_conversation", "memories": []}

    transcript = "\n".join(
        f"{'User' if r.role == 'user' else 'Clara'}: {r.content}" for r in rows
    )

    if chat_completion is None:
        from app.services.clara_conversation_service import ClaraConversationService
        chat_completion = ClaraConversationService()._chat_completion

    from app.services.clara_conversation_service import CLARA_MODEL

    response = await chat_completion(
        model=CLARA_MODEL,
        reasoning_effort="low",
        messages=[{
            "role": "user",
            "content": _PROMPT.format(
                max_chars=MAX_MEMORY_CHARS, day=target_date.isoformat(), transcript=transcript
            ),
        }],
        max_tokens=2000,
        temperature=0.3,
    )
    statements = parse_memory_statements(response.choices[0].message.content)
    if not statements:
        return {"user_id": user_id, "skipped": "nothing_worth_remembering", "memories": []}

    # One batched call for the whole night's statements. None => store unembedded
    # rows; the backfill script and the keyword path both still cover them.
    from app.services.embeddings import embed_texts
    vectors = await embed_texts(statements)
    if vectors is None or len(vectors) != len(statements):
        vectors = [None] * len(statements)

    existing = list((await session.execute(
        select(ConversationLog.content).where(
            ConversationLog.user_id == user_id,
            ConversationLog.role == MEMORY_ROLE,
        )
    )).scalars().all())

    kept = []
    for statement, vector in zip(statements, vectors):
        if _is_near_duplicate(statement, existing):
            continue
        if vector is not None and await _has_similar_memory(session, user_id, vector):
            continue
        existing.append(statement)
        kept.append(statement)
        session.add(ConversationLog(
            user_id=user_id,
            conversation_id=memory_conversation_id(target_date),
            role=MEMORY_ROLE,
            content=statement,
            embedding=vector,
            meta={"source_date": target_date.isoformat()},
        ))

    await session.commit()
    logger.info("Consolidated %s memories for user %s on %s", len(kept), user_id, target_date)
    return {
        "user_id": user_id,
        "memories": kept,
        "dropped_as_duplicate": len(statements) - len(kept),
    }


async def consolidate_day(target_date: date, user_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """Every user who talked to Clara on target_date."""
    from app.core.database import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        if user_ids is None:
            start, end = _day_bounds(target_date)
            user_ids = list((await session.execute(
                select(distinct(ConversationLog.user_id)).where(
                    ConversationLog.role.in_(("user", "assistant")),
                    ConversationLog.created_at >= start,
                    ConversationLog.created_at < end,
                )
            )).scalars().all())

        results = []
        for i, user_id in enumerate(user_ids):
            if i:
                await asyncio.sleep(PACING_SECONDS)  # free-tier RPM
            try:
                results.append(await consolidate_user_day(session, user_id, target_date))
            except Exception as e:
                logger.error("Consolidation failed for user %s: %s", user_id, e, exc_info=True)
                await session.rollback()
                results.append({"user_id": user_id, "error": str(e), "memories": []})

    return {
        "target_date": target_date.isoformat(),
        "users": len(user_ids),
        "memories_written": sum(len(r["memories"]) for r in results),
        "results": results,
    }


@celery_app.task(bind=True, name="memory.consolidate_daily")
def consolidate_daily_memories(self, target_date_str: Optional[str] = None) -> Dict[str, Any]:
    """Distil yesterday's conversations into durable memories. Re-runs are no-ops."""
    target_date = (
        date.fromisoformat(target_date_str) if target_date_str
        else datetime.now(timezone.utc).date() - timedelta(days=1)
    )
    return asyncio.run(consolidate_day(target_date))
