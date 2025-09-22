"""
Conversation Context Configuration for Enhanced Conversational Context Integration.
Centralizes configuration for conversation context parameters as per Story 2.6.
"""
import os
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ConversationContextConfig:
    """Centralized configuration for conversation context parameters"""

    # Simulation Event Context
    RECENT_EVENTS_HOURS_BACK: int = 72
    MAX_EVENTS_COUNT: int = 5

    # Backstory Content Limits
    MAX_BACKSTORY_CHARS: int = 8000  # ~2000 tokens
    MAX_BACKSTORY_TOKENS: int = 2000

    # State Retrieval Windows
    GLOBAL_STATE_CACHE_TTL_MINUTES: int = 5
    SESSION_STATE_CACHE_TTL_MINUTES: int = 30

    # Context Selection Weights
    SIMULATION_EVENTS_WEIGHT: float = 0.6
    BACKSTORY_CONTENT_WEIGHT: float = 0.4

    # Performance Thresholds
    MAX_CONTEXT_PROCESSING_MS: int = 100
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = 3
    CIRCUIT_BREAKER_TIMEOUT_SECONDS: int = 30

    # Content Type Priorities (higher = more important when selecting)
    CONTENT_TYPE_PRIORITIES: Dict[str, int] = None

    def __post_init__(self):
        """Set default values for complex fields"""
        if self.CONTENT_TYPE_PRIORITIES is None:
            self.CONTENT_TYPE_PRIORITIES = {
                "character_gist": 1,
                "childhood_memories": 3,
                "positive_memories": 2,
                "connecting_memories": 4,  # trauma content - highest priority when relevant
                "friend_character": 2
            }

    @classmethod
    def from_env(cls) -> 'ConversationContextConfig':
        """Load configuration from environment variables with defaults"""
        return cls(
            RECENT_EVENTS_HOURS_BACK=int(os.getenv("CONV_RECENT_EVENTS_HOURS", "24")),
            MAX_EVENTS_COUNT=int(os.getenv("CONV_MAX_EVENTS_COUNT", "5")),
            MAX_BACKSTORY_CHARS=int(os.getenv("CONV_MAX_BACKSTORY_CHARS", "8000")),
            MAX_BACKSTORY_TOKENS=int(os.getenv("CONV_MAX_BACKSTORY_TOKENS", "2000")),
            GLOBAL_STATE_CACHE_TTL_MINUTES=int(os.getenv("CONV_GLOBAL_STATE_CACHE_TTL", "5")),
            SESSION_STATE_CACHE_TTL_MINUTES=int(os.getenv("CONV_SESSION_STATE_CACHE_TTL", "30")),
            SIMULATION_EVENTS_WEIGHT=float(os.getenv("CONV_SIMULATION_EVENTS_WEIGHT", "0.6")),
            BACKSTORY_CONTENT_WEIGHT=float(os.getenv("CONV_BACKSTORY_CONTENT_WEIGHT", "0.4")),
            MAX_CONTEXT_PROCESSING_MS=int(os.getenv("CONV_MAX_CONTEXT_PROCESSING_MS", "100")),
            CIRCUIT_BREAKER_FAILURE_THRESHOLD=int(os.getenv("CONV_CIRCUIT_BREAKER_FAILURE_THRESHOLD", "3")),
            CIRCUIT_BREAKER_TIMEOUT_SECONDS=int(os.getenv("CONV_CIRCUIT_BREAKER_TIMEOUT_SECONDS", "30"))
        )


# Global configuration instance
conversation_config = ConversationContextConfig.from_env()