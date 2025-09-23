# Import all models here so Alembic can detect them
from .user import User
from .session import Session, SessionTranscript
from .vocabulary import VocabularyAssessment, VocabularyRecommendation, VocabularyUsage, VocabularySuggestion
from .conversation_v2 import Conversation, ConversationMessage, SessionState, UserPreferences
from .clara_state import ClaraState
from .session_state import UserSessionState, StateChangeHistory, SessionStateBackup
from .subscription import SubscriptionPlan, UserSubscription, PaymentRecord
from .simulation import GlobalEvents, ClaraGlobalState, SimulationLog, SimulationConfig
from .journal import JournalEntries, JournalGenerationLog, JournalTemplate

__all__ = [
    "User",
    "Session",
    "SessionTranscript",
    "VocabularyAssessment",
    "VocabularyRecommendation",
    "VocabularyUsage",
    "VocabularySuggestion",
    "Conversation",
    "ConversationMessage",
    "SessionState",
    "UserPreferences",
    "ClaraState",
    "UserSessionState",
    "StateChangeHistory",
    "SessionStateBackup",
    "SubscriptionPlan",
    "UserSubscription",
    "PaymentRecord",
    "GlobalEvents",
    "ClaraGlobalState",
    "SimulationLog",
    "SimulationConfig",
    "JournalEntries",
    "JournalGenerationLog",
    "JournalTemplate"
]