from sqlalchemy import Integer, String, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import Mapped, mapped_column
from typing import Optional
from app.core.database import Base


class AvaState(Base):
    """
    AvaState model for storing Ava's persistent core traits and attributes.
    
    This table implements the Global State pattern from the dual-state architecture,
    storing persistent attributes that survive across conversation sessions.
    """
    __tablename__ = "ava_state"

    state_id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    trait_name: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    value: Mapped[str] = mapped_column(String, nullable=False)
    last_updated: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(),
        onupdate=func.now()
    )

    def __repr__(self):
        return f"<AvaState(trait_name='{self.trait_name}', value='{self.value}')>"