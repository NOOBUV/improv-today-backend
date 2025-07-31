from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, JSON, ForeignKey, Float
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.core.database import Base

class Session(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Session metadata
    session_type = Column(String, default="practice")  # assessment, practice, daily
    topic = Column(String, nullable=True)
    status = Column(String, default="active")  # active, completed, failed, abandoned
    
    # Timing
    start_time = Column(DateTime(timezone=True), server_default=func.now())
    end_time = Column(DateTime(timezone=True), nullable=True)
    duration_seconds = Column(Integer, nullable=True)
    
    # Metrics
    word_count = Column(Integer, default=0)
    vocabulary_used_count = Column(Integer, default=0)
    fluency_score = Column(Float, nullable=True)
    overall_score = Column(Float, nullable=True)
    
    # Analysis results
    analysis_data = Column(JSON, nullable=True)  # Store detailed analysis results
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    transcripts = relationship("SessionTranscript", back_populates="session", cascade="all, delete-orphan")
    vocabulary_usage = relationship("VocabularyUsage", back_populates="session", cascade="all, delete-orphan")

class SessionTranscript(Base):
    __tablename__ = "session_transcripts"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    
    # Transcript data
    original_text = Column(Text, nullable=True)  # Raw speech-to-text output
    cleaned_text = Column(Text, nullable=False)  # GPT-cleaned transcript
    confidence_score = Column(Float, nullable=True)  # Confidence in cleaning
    corrections_made = Column(JSON, nullable=True)  # List of corrections
    
    # Analysis
    detected_vocabulary_level = Column(String, nullable=True)
    word_complexity_score = Column(Float, nullable=True)
    grammar_score = Column(Float, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    session = relationship("Session", back_populates="transcripts")