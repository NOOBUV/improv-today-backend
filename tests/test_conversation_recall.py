"""
Associative recall: keyword lookup over conversation_log + the prompt section it feeds.
"""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import StaticPool

from app.models.conversation_log import ConversationLog
from app.services.session_state_service import SessionStateService, PastSnippet
from app.services.conversation_prompt_service import ConversationPromptService, EmotionType


async def _seeded_maker(monkeypatch, rows):
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:", poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(ConversationLog.__table__.create)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as session:
        session.add_all(rows)
        await session.commit()
    monkeypatch.setattr("app.core.database.AsyncSessionLocal", maker)
    return engine


@pytest.mark.asyncio
async def test_recall_finds_past_conversation_and_excludes_current(monkeypatch):
    now = datetime.now(timezone.utc)
    engine = await _seeded_maker(monkeypatch, [
        ConversationLog(
            user_id="42", conversation_id="old-1", role="user",
            content="that networking event was a disaster, I knew nobody there",
            created_at=now - timedelta(days=2),
        ),
        ConversationLog(
            user_id="42", conversation_id="old-1", role="assistant",
            content="totally unrelated line about laundry",
            created_at=now - timedelta(days=2),
        ),
        ConversationLog(
            user_id="42", conversation_id="current", role="user",
            content="the networking thing again",
            created_at=now,
        ),
        ConversationLog(
            user_id="99", conversation_id="other-user", role="user",
            content="networking event for somebody else entirely",
            created_at=now - timedelta(days=1),
        ),
    ])

    snippets = await SessionStateService().get_related_past_snippets(
        user_id="42",
        user_message="remember that networking event?",
        exclude_conversation_id="current",
    )

    assert [s.content for s in snippets] == [
        "that networking event was a disaster, I knew nobody there"
    ]
    assert snippets[0].role == "user"
    assert snippets[0].age == "2 days ago"
    await engine.dispose()


@pytest.mark.asyncio
async def test_recall_returns_empty_when_nothing_matches(monkeypatch):
    engine = await _seeded_maker(monkeypatch, [
        ConversationLog(user_id="42", conversation_id="old-1", role="user",
                        content="something about sourdough"),
    ])
    assert await SessionStateService().get_related_past_snippets(
        user_id="42", user_message="how was the pitch?", exclude_conversation_id="current",
    ) == []
    await engine.dispose()


@pytest.mark.asyncio
async def test_recall_is_best_effort(monkeypatch):
    def boom():
        raise RuntimeError("db down")
    monkeypatch.setattr("app.core.database.AsyncSessionLocal", boom)
    assert await SessionStateService().get_related_past_snippets(
        user_id="42", user_message="networking event", exclude_conversation_id="c",
    ) == []


def _prompt(**kwargs):
    return ConversationPromptService().construct_conversation_prompt_with_mood(
        character_backstory="Clara, 22.",
        user_message="remember that networking event?",
        conversation_emotion=EmotionType.CALM,
        **kwargs,
    )


def test_memory_section_omitted_when_no_memories():
    assert "THINGS YOU REMEMBER" not in _prompt()
    assert "THINGS YOU REMEMBER" not in _prompt(past_memories=[])


def test_memory_section_rendered_per_speaker():
    prompt = _prompt(past_memories=[
        PastSnippet(role="user", content="the networking event was a disaster", age="2 days ago"),
        PastSnippet(role="assistant", content="I hid by the snack table", age="yesterday"),
    ])
    assert "THINGS YOU REMEMBER FROM PAST CONVERSATIONS:" in prompt
    assert '- 2 days ago, they said: "the networking event was a disaster"' in prompt
    assert '- yesterday, you told them: "I hid by the snack table"' in prompt
    assert "only if it genuinely fits" in prompt
