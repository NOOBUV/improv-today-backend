# Import all models here so Alembic can detect them
from .user import User
from .session import Session, SessionTranscript
from .vocabulary import VocabularyAssessment, VocabularyRecommendation, VocabularyUsage

__all__ = [
    "User",
    "Session", 
    "SessionTranscript",
    "VocabularyAssessment",
    "VocabularyRecommendation", 
    "VocabularyUsage"
]