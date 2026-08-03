"""
Append-only durable transcript of every conversation turn.

Session state lives in Redis with a 24h TTL (SessionStateService), so it is
ephemeral. This table is the permanent record: one row per user/assistant
message, never updated, never deleted.
"""

from sqlalchemy import Integer, String, Text, DateTime, JSON, LargeBinary, cast
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import UserDefinedType
from typing import Optional, Dict, Any, List
from datetime import datetime

from app.core.database import Base


class Vector(UserDefinedType):
    """Minimal pgvector column type.

    ponytail: ~10 lines instead of the pgvector package. It exists so alembic
    autogenerate doesn't propose dropping the column and so the ORM can write a
    list[float]; every vector *query* is raw SQL on the postgres-only path.
    """

    cache_ok = True

    def __init__(self, dim: int):
        self.dim = dim

    def get_col_spec(self, **kw) -> str:
        return f"vector({self.dim})"

    def bind_processor(self, dialect):
        def process(value):
            if value is None:
                return None
            return "[" + ",".join(f"{x:.7g}" for x in value) + "]"
        return process

    def bind_expression(self, bindvalue):
        # psycopg sends the processed value as text; postgres will not implicitly
        # coerce text into vector, so the cast has to be in the SQL itself.
        return cast(bindvalue, self)


class ConversationLog(Base):
    """One conversation turn (user utterance or Clara reply)."""

    __tablename__ = "conversation_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    conversation_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # "user" | "assistant" | "memory"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # Nullable: an un-embedded row is simply one only the keyword path can find.
    # ponytail: LargeBinary on sqlite so tests can create_all — the vector path is
    # postgres-only and never touches the column under sqlite.
    embedding: Mapped[Optional[List[float]]] = mapped_column(
        Vector(768).with_variant(LargeBinary(), "sqlite"), nullable=True
    )
    # ponytail: JSONB on postgres, plain JSON on sqlite so tests can create_all
    meta: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB().with_variant(JSON(), "sqlite"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
