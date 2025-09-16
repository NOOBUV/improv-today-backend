"""
Session state models for per-user conversation context.
Implements session-specific state management with Redis and database backing.
"""

from sqlalchemy import Integer, String, DateTime, Text, JSON, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing import Optional, Dict, Any
from app.core.database import Base


class UserSessionState(Base):
    """
    UserSessionState model for storing per-user conversation data.

    This model stores session-specific state adjustments and conversation context
    that are isolated per user and conversation session. Used as backup storage
    for Redis-based session management.
    """
    __tablename__ = "user_session_states"

    session_state_id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_id: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    user_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    conversation_id: Mapped[str] = mapped_column(String, index=True, nullable=False)

    # Global state baseline snapshot
    global_state_baseline: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)

    # Session-specific adjustments
    session_adjustments: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)

    # User personalization data
    personalization_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)

    # Conversation context
    conversation_context: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)

    # Session metadata
    session_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)

    # Status and lifecycle
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    last_activity: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )

    # Timestamps
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )
    updated_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )
    expires_at: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True), nullable=True)

    def __repr__(self):
        return f"<UserSessionState(session_id='{self.session_id}', user_id='{self.user_id}', active={self.is_active})>"


class StateChangeHistory(Base):
    """
    StateChangeHistory model for audit trail and time-series analytics.

    This model tracks all state changes over time for analysis and debugging,
    providing a complete audit trail of how Ava's state evolves based on
    events and emotional processing.
    """
    __tablename__ = "state_change_history"

    history_id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    # Source information
    event_id: Mapped[Optional[str]] = mapped_column(String, index=True, nullable=True)
    session_id: Mapped[Optional[str]] = mapped_column(String, index=True, nullable=True)
    user_id: Mapped[Optional[str]] = mapped_column(String, index=True, nullable=True)

    # Change details
    change_type: Mapped[str] = mapped_column(String, index=True, nullable=False)  # event_impact, emotional_processing, manual_adjustment
    trait_name: Mapped[str] = mapped_column(String, index=True, nullable=False)

    # State values
    previous_value: Mapped[Optional[float]] = mapped_column(nullable=True)
    new_value: Mapped[float] = mapped_column(nullable=False)
    change_amount: Mapped[float] = mapped_column(nullable=False)

    # Context and reasoning
    change_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    context_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)

    # Processing metadata
    processing_source: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # state_manager, consciousness_generator, session_service
    confidence_score: Mapped[Optional[float]] = mapped_column(nullable=True)  # 0-1 confidence in change accuracy

    # Timestamps
    timestamp: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        index=True
    )
    processed_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )

    def __repr__(self):
        return f"<StateChangeHistory(trait='{self.trait_name}', change={self.change_amount:+.1f}, type='{self.change_type}')>"


class SessionStateBackup(Base):
    """
    SessionStateBackup model for persistent storage of session data.

    This model provides database backup for Redis-based session storage,
    enabling recovery and long-term analytics of session patterns.
    """
    __tablename__ = "session_state_backups"

    backup_id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    user_id: Mapped[str] = mapped_column(String, index=True, nullable=False)

    # Complete session state snapshot
    session_data: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)

    # Backup metadata
    backup_reason: Mapped[str] = mapped_column(String, nullable=False)  # scheduled, manual, expiration, error_recovery
    redis_available: Mapped[bool] = mapped_column(default=True, nullable=False)

    # Timestamps
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        index=True
    )

    def __repr__(self):
        return f"<SessionStateBackup(session_id='{self.session_id}', reason='{self.backup_reason}')>"