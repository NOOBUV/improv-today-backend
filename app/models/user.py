from sqlalchemy import Integer, String, DateTime, Boolean, Text, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing import Optional, List, TYPE_CHECKING
from app.core.database import Base

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    # Anonymous identity cookie UUID
    anon_uuid: Mapped[Optional[str]] = mapped_column(String, unique=True, index=True, nullable=True)
    email: Mapped[Optional[str]] = mapped_column(String, unique=True, index=True, nullable=True)
    hashed_password: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_anonymous: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Vocabulary tier and assessment data
    vocabulary_tier: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    assessment_completed: Mapped[bool] = mapped_column(Boolean, default=False)
    interests: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    
    # Timestamps
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    sessions: Mapped[List["Session"]] = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    vocabulary_assessments: Mapped[List["VocabularyAssessment"]] = relationship("VocabularyAssessment", back_populates="user", cascade="all, delete-orphan")
    vocabulary_recommendations: Mapped[List["VocabularyRecommendation"]] = relationship("VocabularyRecommendation", back_populates="user", cascade="all, delete-orphan")

if TYPE_CHECKING:
    from app.models.session import Session  # noqa: F401
    from app.models.vocabulary import VocabularyAssessment, VocabularyRecommendation  # noqa: F401