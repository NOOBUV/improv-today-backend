"""
New conversation models for the redesigned architecture.
These models follow the system redesign plan with UUID primary keys,
proper message tracking, and real-time state management.
"""
from sqlalchemy import Column, String, DateTime, Text, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.core.database import Base
import uuid


class Conversation(Base):
    """
    Main conversation entity following the redesign plan.
    Replaces the session-centric approach with conversation-centric design.
    """
    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(String(255), index=True, nullable=True)  # Support anonymous users
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=True)  # Migration link
    status = Column(String(50), default='active')  # active, completed, paused, ended
    personality = Column(String(50), default='friendly_neutral')
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    conversation_metadata = Column(JSON, nullable=True)  # Store conversation metadata
    
    # Relationships
    messages = relationship("ConversationMessage", back_populates="conversation", cascade="all, delete-orphan")
    session_state = relationship("SessionState", back_populates="conversation", uselist=False, cascade="all, delete-orphan")
    legacy_session = relationship("Session", foreign_keys=[session_id])  # For migration


class ConversationMessage(Base):
    """
    Individual messages within a conversation (user and AI responses).
    Provides granular tracking of conversation flow.
    """
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False, index=True)
    role = Column(String(20), nullable=False, index=True)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    audio_url = Column(String(500), nullable=True)  # For TTS audio storage
    feedback = Column(JSON, nullable=True)  # Feedback data for the message
    processing_time = Column(Integer, nullable=True)  # AI response time in milliseconds
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")


class SessionState(Base):
    """
    Real-time session state for WebSocket synchronization.
    Prevents race conditions and maintains conversation flow.
    """
    __tablename__ = "session_states"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False, unique=True, index=True)
    state = Column(String(50), nullable=False, index=True)  # conversation state machine states
    transcript = Column(Text, nullable=True)  # Current finalized transcript
    interim_transcript = Column(Text, nullable=True)  # Interim speech recognition results
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), index=True)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="session_state")


class UserPreferences(Base):
    """
    User preferences for voice, personality, and other settings.
    Enables personalized conversation experiences.
    """
    __tablename__ = "user_preferences"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(String(255), nullable=False, unique=True, index=True)
    preferred_voice = Column(String(100), nullable=True)
    preferred_personality = Column(String(50), default='friendly_neutral')
    settings = Column(JSON, nullable=True)  # Additional user settings
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())