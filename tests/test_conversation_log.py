"""
Durable conversation transcript: write path (SessionStateService) + read endpoint.
"""

import pytest
from unittest.mock import AsyncMock, Mock

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy import select

from app.main import app as fastapi_app
from app.core.database import get_db
from app.auth.dependencies import get_current_user
from app.models.conversation_log import ConversationLog
from app.models.user import User
from app.services.session_state_service import SessionStateService


@pytest.mark.asyncio
async def test_add_conversation_message_writes_log_row(monkeypatch):
    """Redis write and durable log write both happen at the same choke point."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:", poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(ConversationLog.__table__.create)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr("app.core.database.AsyncSessionLocal", maker)

    service = SessionStateService()
    service.redis_service = Mock()
    service.get_session_state = AsyncMock(return_value={
        "conversation_messages": [],
        "session_metadata": {"total_interactions": 0},
    })
    service._store_session_state = AsyncMock(return_value=True)

    assert await service.add_conversation_message(
        user_id="42", conversation_id="conv-1",
        message_type="user", message_content="hello clara",
    )
    assert await service.add_conversation_message(
        user_id="42", conversation_id="conv-1",
        message_type="assistant", message_content="hi there",
        metadata={"conversation_emotion": "happy"},
    )

    async with maker() as session:
        rows = (await session.execute(
            select(ConversationLog).order_by(ConversationLog.id)
        )).scalars().all()

    assert [(r.role, r.content) for r in rows] == [
        ("user", "hello clara"), ("assistant", "hi there")
    ]
    assert rows[0].meta is None
    assert rows[1].meta == {"conversation_emotion": "happy"}
    await engine.dispose()


@pytest.mark.asyncio
async def test_log_write_failure_does_not_fail_the_turn(monkeypatch):
    """Best-effort: a broken log write must not break the conversation."""
    def boom():
        raise RuntimeError("db down")
    monkeypatch.setattr("app.core.database.AsyncSessionLocal", boom)

    service = SessionStateService()
    service.redis_service = Mock()
    service.get_session_state = AsyncMock(return_value={
        "conversation_messages": [], "session_metadata": {"total_interactions": 0},
    })
    service._store_session_state = AsyncMock(return_value=True)

    assert await service.add_conversation_message(
        user_id="42", conversation_id="conv-1",
        message_type="user", message_content="still works",
    )


def test_conversation_log_endpoint_returns_current_user_rows():
    engine = create_engine(
        "sqlite:///:memory:", poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    ConversationLog.__table__.create(engine)
    db = sessionmaker(bind=engine)()
    db.add_all([
        ConversationLog(user_id="7", conversation_id="c1", role="user", content="mine"),
        ConversationLog(user_id="7", conversation_id="c2", role="assistant", content="other conv"),
        ConversationLog(user_id="8", conversation_id="c1", role="user", content="someone else"),
    ])
    db.commit()

    fastapi_app.dependency_overrides[get_db] = lambda: db
    fastapi_app.dependency_overrides[get_current_user] = lambda: User(id=7)
    try:
        client = TestClient(fastapi_app)

        rows = client.get("/api/clara/conversation/log").json()
        assert {r["content"] for r in rows} == {"mine", "other conv"}

        scoped = client.get("/api/clara/conversation/log?conversation_id=c1").json()
        assert [r["content"] for r in scoped] == ["mine"]
    finally:
        fastapi_app.dependency_overrides.clear()
        db.close()
