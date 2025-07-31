from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, JSON, ForeignKey, Float
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.core.database import Base

class VocabularyAssessment(Base):
    __tablename__ = "vocabulary_assessments"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Assessment results
    vocabulary_tier = Column(String, nullable=False)  # beginner, intermediate, advanced
    complexity_score = Column(Float, nullable=True)  # 0-100 complexity rating
    
    # Analysis data
    strengths = Column(JSON, nullable=True)  # Areas where user performs well
    gaps = Column(JSON, nullable=True)  # Areas needing improvement
    interests = Column(JSON, nullable=True)  # Topics user is interested in
    sample_topics = Column(JSON, nullable=True)  # Topics discussed during assessment
    
    # Assessment metadata
    word_count = Column(Integer, default=0)
    unique_words_used = Column(Integer, default=0)
    assessment_duration = Column(Integer, nullable=True)  # Duration in seconds
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="vocabulary_assessments")

class VocabularyRecommendation(Base):
    __tablename__ = "vocabulary_recommendations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Word details
    word = Column(String, nullable=False, index=True)
    definition = Column(Text, nullable=True)
    context_example = Column(Text, nullable=True)  # Example sentence
    difficulty_level = Column(String, nullable=False)  # beginner, intermediate, advanced
    category = Column(String, nullable=True)  # topic category (work, hobbies, etc.)
    
    # Recommendation metadata
    priority_score = Column(Float, default=0.0)  # Higher = more important to learn
    based_on_interests = Column(Boolean, default=False)
    based_on_gaps = Column(Boolean, default=False)
    
    # Usage tracking
    status = Column(String, default="pending")  # pending, practicing, mastered, skipped
    times_encountered = Column(Integer, default=0)
    times_used_correctly = Column(Integer, default=0)
    last_practiced = Column(DateTime(timezone=True), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="vocabulary_recommendations")

class VocabularyUsage(Base):
    __tablename__ = "vocabulary_usage"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    
    # Word usage details
    word = Column(String, nullable=False, index=True)
    used_correctly = Column(Boolean, nullable=False)
    context_sentence = Column(Text, nullable=True)  # Sentence where word was used
    
    # Scoring
    usage_score = Column(Float, nullable=True)  # 0-100 score for how well word was used
    appropriateness_score = Column(Float, nullable=True)  # How appropriate for context
    
    # Analysis
    feedback = Column(Text, nullable=True)  # GPT feedback on usage
    suggested_improvement = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    session = relationship("Session", back_populates="vocabulary_usage")