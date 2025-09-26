from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.core.database import Base

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String, primary_key=True, index=True)  # UUID as string
    user_id = Column(String, ForeignKey("users.id"))  # String to match user_id type
    session_id = Column(Integer, index=True)  # Session tracking
    status = Column(String, default='active', index=True)  # Conversation status
    personality = Column(String)  # Personality setting
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    conversation_metadata = Column(JSON)  # Additional metadata

    user = relationship("User", back_populates="conversations")