from sqlalchemy import Integer, String, DateTime, Float, Text
from sqlalchemy.sql import func
from sqlalchemy.orm import Mapped, mapped_column
from typing import Optional
from app.core.database import Base


class AvaState(Base):
    """
    Enhanced AvaState model for storing Ava's persistent core traits and attributes.

    This table implements the Global State pattern from the dual-state architecture,
    storing persistent attributes that survive across conversation sessions.
    Enhanced with numeric values, trend tracking, and change metadata.
    """
    __tablename__ = "ava_state"

    state_id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    trait_name: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    value: Mapped[str] = mapped_column(String, nullable=False)

    # Enhanced tracking fields
    numeric_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    trend: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # increasing, decreasing, stable
    change_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    last_event_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # Bounds for validation
    min_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    max_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Timestamps
    last_updated: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )

    def __repr__(self):
        return f"<AvaState(trait_name='{self.trait_name}', value='{self.value}', numeric_value={self.numeric_value})>"