"""
Nightly consolidation: parsing, idempotency and dedupe.

sqlite here, so there is no pgvector — embed_texts is stubbed to None, which is
the same "embeddings unavailable" path the job takes in production when the
quota is gone. The cosine half of dedupe is tested by stubbing _has_similar_memory.
"""

from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import StaticPool

from app.models.conversation_log import ConversationLog
from app.services import memory_consolidation as mc

TARGET = date(2026, 8, 2)


def _reply(text):
    """Shape of an OpenAI-style chat completion, which is all the job touches."""
    async def chat_completion(**kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
        )
    return chat_completion


async def _session_with_day(monkeypatch, extra_rows=()):
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:", poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(ConversationLog.__table__.create)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    at = datetime(2026, 8, 2, 12, 0, tzinfo=timezone.utc)
    turns = [
        ConversationLog(user_id="42", conversation_id="c1", role=role,
                        content=content, created_at=at)
        for role, content in [
            ("user", "my three week project got cancelled out of nowhere today"),
            ("assistant", "that's brutal. how are you holding up?"),
            ("user", "honestly rough. my manager Dana told me an hour before standup"),
            ("assistant", "an hour. that's not a heads up, that's an ambush."),
        ]
    ]
    async with maker() as s:
        s.add_all(turns + list(extra_rows))
        await s.commit()

    monkeypatch.setattr("app.core.database.AsyncSessionLocal", maker)

    async def no_embeddings(_texts):
        return None
    monkeypatch.setattr("app.services.embeddings.embed_texts", no_embeddings)
    return engine, maker


@pytest.mark.asyncio
async def test_consolidation_writes_memory_rows(monkeypatch):
    engine, maker = await _session_with_day(monkeypatch)
    async with maker() as s:
        result = await mc.consolidate_user_day(
            s, "42", TARGET,
            chat_completion=_reply(
                '["The user\'s three-week project was cancelled without warning.",'
                ' "The user\'s manager is named Dana."]'
            ),
        )

    assert len(result["memories"]) == 2
    async with maker() as s:
        rows = (await s.execute(
            select(ConversationLog).where(ConversationLog.role == "memory")
        )).scalars().all()
    assert len(rows) == 2
    assert all(r.conversation_id == "consolidated:2026-08-02" for r in rows)
    assert all(r.meta == {"source_date": "2026-08-02"} for r in rows)
    await engine.dispose()


@pytest.mark.asyncio
async def test_consolidation_is_idempotent(monkeypatch):
    """Second run for the same day writes nothing — beat retries must be free."""
    engine, maker = await _session_with_day(monkeypatch)
    reply = _reply('["The user\'s manager is named Dana."]')

    async with maker() as s:
        first = await mc.consolidate_user_day(s, "42", TARGET, chat_completion=reply)
    async with maker() as s:
        second = await mc.consolidate_user_day(s, "42", TARGET, chat_completion=reply)

    assert len(first["memories"]) == 1
    assert second["skipped"] == "already_consolidated"
    assert second["memories"] == []

    async with maker() as s:
        count = len((await s.execute(
            select(ConversationLog).where(ConversationLog.role == "memory")
        )).scalars().all())
    assert count == 1
    await engine.dispose()


@pytest.mark.asyncio
async def test_consolidation_drops_restatements_of_known_memories(monkeypatch):
    """A fact already remembered from an earlier day is not learned twice."""
    engine, maker = await _session_with_day(monkeypatch, extra_rows=[
        ConversationLog(
            user_id="42", conversation_id="consolidated:2026-08-01", role="memory",
            content="The user's manager is named Dana.",
            created_at=datetime(2026, 8, 1, 23, 0, tzinfo=timezone.utc),
        ),
    ])

    async with maker() as s:
        result = await mc.consolidate_user_day(
            s, "42", TARGET,
            chat_completion=_reply(
                '["The user\'s manager is named Dana",'
                ' "The user\'s three-week project was cancelled without warning."]'
            ),
        )

    assert result["memories"] == [
        "The user's three-week project was cancelled without warning."
    ]
    assert result["dropped_as_duplicate"] == 1
    await engine.dispose()


@pytest.mark.asyncio
async def test_consolidation_drops_semantic_duplicates(monkeypatch):
    """Different words, same fact — caught by cosine, not by string overlap."""
    engine, maker = await _session_with_day(monkeypatch)

    async def vectors(texts):
        return [[0.1] * 768 for _ in texts]
    monkeypatch.setattr("app.services.embeddings.embed_texts", vectors)

    seen = []

    async def already_have_it(session, user_id, vector):
        seen.append(user_id)
        return True
    monkeypatch.setattr(mc, "_has_similar_memory", already_have_it)

    async with maker() as s:
        result = await mc.consolidate_user_day(
            s, "42", TARGET,
            chat_completion=_reply('["Dana, the user\'s boss, runs their standup."]'),
        )

    assert result["memories"] == []
    assert result["dropped_as_duplicate"] == 1
    assert seen == ["42"]
    await engine.dispose()


@pytest.mark.asyncio
async def test_consolidation_skips_days_with_nothing_to_sleep_on(monkeypatch):
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:", poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(ConversationLog.__table__.create)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as s:
        s.add(ConversationLog(
            user_id="42", conversation_id="c1", role="user", content="hey",
            created_at=datetime(2026, 8, 2, 9, 0, tzinfo=timezone.utc),
        ))
        await s.commit()

    async def explode(**kwargs):
        raise AssertionError("should not call the LLM for a one-line day")

    async with maker() as s:
        result = await mc.consolidate_user_day(s, "42", TARGET, chat_completion=explode)
    assert result["skipped"] == "too_little_conversation"
    await engine.dispose()


@pytest.mark.parametrize("raw,expected", [
    ('["one statement that is long enough", "another one here"]',
     ["one statement that is long enough", "another one here"]),
    ('```json\n["fenced statement about the user"]\n```',
     ["fenced statement about the user"]),
    ('Here you go:\n["a statement about the user here"]',
     ["a statement about the user here"]),
    # The model ignoring the JSON envelope entirely is the case this exists for.
    ("- The user dislikes mondays quite a lot\n- The user's cat is called Pepper",
     ["The user dislikes mondays quite a lot", "The user's cat is called Pepper"]),
    ("", []),
    (None, []),
])
def test_parse_memory_statements(raw, expected):
    assert mc.parse_memory_statements(raw) == expected


def test_parse_memory_statements_caps_length_and_count():
    long_one = "x" * 400
    parsed = mc.parse_memory_statements(
        "[" + ",".join(f'"{long_one}"' for _ in range(12)) + "]"
    )
    assert len(parsed) == mc.MAX_MEMORIES
    assert all(len(p) <= mc.MAX_MEMORY_CHARS for p in parsed)
