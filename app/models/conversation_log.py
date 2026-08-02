"""
Append-only durable transcript of every conversation turn.

Session state lives in Redis with a 24h TTL (SessionStateService), so it is
ephemeral. This table is the permanent record: one row per user/assistant
message, never updated, never deleted.
"""

from sqlalchemy import Integer, String, Text, DateTime, JSON
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from sqlalchemy.orm import Mapped, mapped_column
from typing import Optional, Dict, Any
from datetime import datetime

from app.core.database import Base


class ConversationLog(Base):
    """One conversation turn (user utterance or Clara reply)."""

    __tablename__ = "conversation_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    conversation_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # ponytail: JSONB on postgres, plain JSON on sqlite so tests can create_all
    meta: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB().with_variant(JSON(), "sqlite"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
