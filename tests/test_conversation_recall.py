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


@pytest.mark.asyncio
async def test_embedding_failure_falls_back_to_keyword(monkeypatch):
    """No embedding (no key, quota, outage) → the keyword path still answers.

    This is also the sqlite path: there is no pgvector here, so every other test
    in this file is implicitly exercising the fallback too.
    """
    now = datetime.now(timezone.utc)
    engine = await _seeded_maker(monkeypatch, [
        ConversationLog(
            user_id="42", conversation_id="old-1", role="user",
            content="that networking event was a disaster, I knew nobody there",
            created_at=now - timedelta(days=2),
        ),
    ])

    async def no_embedding(_text):
        return None
    monkeypatch.setattr("app.services.embeddings.embed_one", no_embedding)

    snippets = await SessionStateService().get_related_past_snippets(
        user_id="42",
        user_message="remember that networking event?",
        exclude_conversation_id="current",
    )

    assert [s.content for s in snippets] == [
        "that networking event was a disaster, I knew nobody there"
    ]
    await engine.dispose()


class _CapturingSession:
    """Stands in for an AsyncSession so the vector SQL can be inspected.

    pgvector doesn't exist under sqlite, so the query itself can't run here —
    what this guards is that the query keeps ASKING for the right things.
    """

    def __init__(self, rows):
        self._rows = rows
        self.params = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def execute(self, stmt, params=None):
        self.params = params
        self.sql = " ".join(str(stmt).split())

        class Result:
            def __init__(self, rows): self._rows = rows
            def all(self): return self._rows
        return Result(self._rows)


@pytest.mark.asyncio
async def test_vector_search_excludes_current_conversation_and_applies_floor(monkeypatch):
    from types import SimpleNamespace
    from app.services.session_state_service import (
        RECALL_SIMILARITY_FLOOR, MIN_RECALL_CHARS, MEMORY_SIMILARITY_BOOST,
    )

    async def fake_embed(_text):
        return [0.1] * 768
    monkeypatch.setattr("app.services.embeddings.embed_one", fake_embed)

    session = _CapturingSession([
        SimpleNamespace(
            role="user", content="x" * 250,
            created_at=datetime.now(timezone.utc) - timedelta(days=1), score=0.81,
        ),
    ])
    monkeypatch.setattr("app.core.database.AsyncSessionLocal", lambda: session)

    snippets = await SessionStateService()._vector_past_snippets(
        "42", "how did the pitch go?", "current-convo", 3
    )

    assert session.params["exclude_id"] == "current-convo"
    assert session.params["user_id"] == "42"
    assert session.params["floor"] == RECALL_SIMILARITY_FLOOR
    assert session.params["min_chars"] == MIN_RECALL_CHARS
    assert session.params["memory_boost"] == MEMORY_SIMILARITY_BOOST
    # The floor and the exclusion must be enforced in SQL, not after the LIMIT.
    assert "conversation_id != :exclude_id" in session.sql
    assert "score >= :floor" in session.sql
    assert "LIMIT :limit" in session.sql

    assert len(snippets) == 1
    assert snippets[0].content.endswith("...")  # truncated at _SNIPPET_CHARS
    assert snippets[0].age == "yesterday"


@pytest.mark.asyncio
async def test_vector_path_returns_none_so_caller_falls_back(monkeypatch):
    """None, not [], means "couldn't run" — [] would suppress the keyword path."""
    async def fake_embed(_text):
        return [0.1] * 768
    monkeypatch.setattr("app.services.embeddings.embed_one", fake_embed)

    def boom():
        raise RuntimeError("no pgvector here")
    monkeypatch.setattr("app.core.database.AsyncSessionLocal", boom)

    assert await SessionStateService()._vector_past_snippets(
        "42", "anything", "current", 3
    ) is None


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


def test_consolidated_memory_renders_without_speaker_or_quotes():
    """A nightly memory was never *said* by either of them, so it reads differently."""
    prompt = _prompt(past_memories=[
        PastSnippet(
            role="memory",
            content="The user's three-week project was cancelled without warning.",
            age="yesterday",
        ),
        PastSnippet(role="user", content="I hate mondays", age="yesterday"),
    ])
    assert "- you remember: The user's three-week project was cancelled without warning." in prompt
    assert '"The user\'s three-week project was cancelled' not in prompt  # not quoted
    assert "yesterday, they said" not in prompt.split("- you remember")[0].split(
        "THINGS YOU REMEMBER")[1]  # memory line carries no timestamp
    assert '- yesterday, they said: "I hate mondays"' in prompt  # others unchanged
