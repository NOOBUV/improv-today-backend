# Import all models here so Alembic can detect them
from .user import User
from .session import Session, SessionTranscript
from .vocabulary import VocabularyAssessment, VocabularyRecommendation, VocabularyUsage
from .conversation_v2 import Conversation, ConversationMessage, SessionState, UserPreferences

__all__ = [
    "User",
    "Session", 
    "SessionTranscript",
    "VocabularyAssessment",
    "VocabularyRecommendation", 
    "VocabularyUsage",
    "Conversation",
    "ConversationMessage", 
    "SessionState",
    "UserPreferences"
]