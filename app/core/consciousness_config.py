"""
Enhanced Consciousness Configuration for 2025 prompt engineering techniques.
Manages Chain-of-Thought depth, Few-Shot examples, Constitutional AI rules, and mood influence weights.
"""

import logging
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from app.core.config import settings

logger = logging.getLogger(__name__)


class ConsciousnessLevel(str, Enum):
    """Levels of consciousness generation complexity."""
    BASIC = "basic"          # Simple prompt without enhancements
    ENHANCED = "enhanced"    # Full 2025 prompt engineering techniques
    FALLBACK = "fallback"    # Degraded mode for performance/reliability


@dataclass
class ChainOfThoughtConfig:
    """Configuration for Chain-of-Thought reasoning parameters."""
    enabled: bool = True
    steps_count: int = 6
    step_descriptions: List[str] = field(default_factory=lambda: [
        "Analyze the Event: What exactly happened and how significant is it?",
        "Consider Current State: How do my current mood, energy, and stress levels influence my reaction?",
        "Reflect on Personality: How does this align with my core values, past experiences, and typical responses?",
        "Determine Emotional Impact: What emotions would I genuinely feel in this moment?",
        "Choose Authentic Action: What would I realistically do next, considering my personality and state?",
        "Generate Internal Thoughts: What would actually go through my mind privately?"
    ])
    max_reasoning_length: int = 400  # Maximum characters for reasoning steps output


@dataclass
class FewShotConfig:
    """Configuration for Few-Shot learning examples."""
    enabled: bool = True
    example_count: int = 3
    include_work_examples: bool = True
    include_social_examples: bool = True
    include_personal_examples: bool = True
    max_examples_per_category: int = 2


@dataclass
class ConstitutionalAIConfig:
    """Configuration for Constitutional AI principles."""
    enabled: bool = True
    rules: List[str] = field(default_factory=lambda: [
        "Authenticity Over Generic AI: Always respond as Clara specifically, never with generic helpful assistant language",
        "Emotional Depth: Show nuanced, complex emotions rather than simplified reactions",
        "Character Growth: Demonstrate learning from past experiences and personal evolution",
        "Realistic Responses: Actions must be believable and consistent with a 22-year-old's capabilities and circumstances",
        "Personality Consistency: Maintain Clara's core traits while allowing for natural emotional variation",
        "Human Vulnerability: Include moments of uncertainty, contradiction, or emotional complexity",
        "Contextual Awareness: Responses must reflect current life circumstances and relationships"
    ])
    max_rules_count: int = 7


@dataclass
class MoodInfluenceConfig:
    """Configuration for mood influence weights and processing."""
    enabled: bool = True
    global_state_weight: float = 0.60  # 60% from global simulation state
    recent_events_weight: float = 0.25  # 25% from recent events
    conversation_sentiment_weight: float = 0.15  # 15% from conversation sentiment

    # Mood transition thresholds
    significant_shift_threshold: float = 20.0  # Points change to trigger transition
    conversation_impact_threshold: float = 15.0  # Strong conversation influence
    event_reaction_threshold: float = 12.0  # Strong event influence

    # Performance thresholds
    max_processing_time_ms: float = 100.0  # Maximum mood analysis time


@dataclass
class PerformanceConfig:
    """Configuration for performance monitoring and thresholds."""
    max_consciousness_processing_ms: float = 3000.0  # 3 seconds max for consciousness generation
    max_total_response_time_ms: float = 5000.0  # 5 seconds max total response time
    enable_performance_logging: bool = True
    enable_fallback_on_timeout: bool = True

    # Metrics collection
    collect_success_failure_metrics: bool = True
    metrics_retention_hours: int = 24


class ConsciousnessConfig:
    """
    Enhanced consciousness configuration management with environment variable support
    and graceful fallback mechanisms.
    """

    def __init__(self):
        self.consciousness_level = self._get_consciousness_level()
        self.chain_of_thought = self._build_chain_of_thought_config()
        self.few_shot = self._build_few_shot_config()
        self.constitutional_ai = self._build_constitutional_ai_config()
        self.mood_influence = self._build_mood_influence_config()
        self.performance = self._build_performance_config()

        # Runtime state
        self.fallback_mode_active = False
        self.last_fallback_reason: Optional[str] = None

        logger.info(f"ConsciousnessConfig initialized with level: {self.consciousness_level}")

    def _get_consciousness_level(self) -> ConsciousnessLevel:
        """Determine consciousness level from environment variables."""
        level_str = os.getenv("CONSCIOUSNESS_LEVEL", "enhanced").lower()

        try:
            return ConsciousnessLevel(level_str)
        except ValueError:
            logger.warning(f"Invalid consciousness level '{level_str}', defaulting to enhanced")
            return ConsciousnessLevel.ENHANCED

    def _build_chain_of_thought_config(self) -> ChainOfThoughtConfig:
        """Build Chain-of-Thought configuration from environment variables."""
        return ChainOfThoughtConfig(
            enabled=self._get_bool_env("CONSCIOUSNESS_COT_ENABLED", True),
            steps_count=self._get_int_env("CONSCIOUSNESS_COT_STEPS", 6),
            max_reasoning_length=self._get_int_env("CONSCIOUSNESS_COT_MAX_LENGTH", 400)
        )

    def _build_few_shot_config(self) -> FewShotConfig:
        """Build Few-Shot configuration from environment variables."""
        return FewShotConfig(
            enabled=self._get_bool_env("CONSCIOUSNESS_FEWSHOT_ENABLED", True),
            example_count=self._get_int_env("CONSCIOUSNESS_FEWSHOT_COUNT", 3),
            include_work_examples=self._get_bool_env("CONSCIOUSNESS_FEWSHOT_WORK", True),
            include_social_examples=self._get_bool_env("CONSCIOUSNESS_FEWSHOT_SOCIAL", True),
            include_personal_examples=self._get_bool_env("CONSCIOUSNESS_FEWSHOT_PERSONAL", True),
            max_examples_per_category=self._get_int_env("CONSCIOUSNESS_FEWSHOT_MAX_PER_CATEGORY", 2)
        )

    def _build_constitutional_ai_config(self) -> ConstitutionalAIConfig:
        """Build Constitutional AI configuration from environment variables."""
        return ConstitutionalAIConfig(
            enabled=self._get_bool_env("CONSCIOUSNESS_CONSTITUTIONAL_ENABLED", True),
            max_rules_count=self._get_int_env("CONSCIOUSNESS_CONSTITUTIONAL_MAX_RULES", 7)
        )

    def _build_mood_influence_config(self) -> MoodInfluenceConfig:
        """Build mood influence configuration from environment variables."""
        return MoodInfluenceConfig(
            enabled=self._get_bool_env("CONSCIOUSNESS_MOOD_ENABLED", True),
            global_state_weight=self._get_float_env("CONSCIOUSNESS_MOOD_GLOBAL_WEIGHT", 0.60),
            recent_events_weight=self._get_float_env("CONSCIOUSNESS_MOOD_EVENTS_WEIGHT", 0.25),
            conversation_sentiment_weight=self._get_float_env("CONSCIOUSNESS_MOOD_CONVERSATION_WEIGHT", 0.15),
            significant_shift_threshold=self._get_float_env("CONSCIOUSNESS_MOOD_SHIFT_THRESHOLD", 20.0),
            max_processing_time_ms=self._get_float_env("CONSCIOUSNESS_MAX_PROCESSING_MS", 100.0)
        )

    def _build_performance_config(self) -> PerformanceConfig:
        """Build performance configuration from environment variables."""
        return PerformanceConfig(
            max_consciousness_processing_ms=self._get_float_env("CONSCIOUSNESS_MAX_PROCESSING_MS", 3000.0),
            max_total_response_time_ms=self._get_float_env("CONSCIOUSNESS_MAX_TOTAL_TIME_MS", 5000.0),
            enable_performance_logging=self._get_bool_env("CONSCIOUSNESS_PERF_LOGGING", True),
            enable_fallback_on_timeout=self._get_bool_env("CONSCIOUSNESS_FALLBACK_ON_TIMEOUT", True),
            collect_success_failure_metrics=self._get_bool_env("CONSCIOUSNESS_COLLECT_METRICS", True),
            metrics_retention_hours=self._get_int_env("CONSCIOUSNESS_METRICS_RETENTION_HOURS", 24)
        )

    def _get_bool_env(self, key: str, default: bool) -> bool:
        """Get boolean from environment variable with default."""
        value = os.getenv(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on')

    def _get_int_env(self, key: str, default: int) -> int:
        """Get integer from environment variable with default."""
        try:
            return int(os.getenv(key, str(default)))
        except (ValueError, TypeError):
            logger.warning(f"Invalid integer value for {key}, using default: {default}")
            return default

    def _get_float_env(self, key: str, default: float) -> float:
        """Get float from environment variable with default."""
        try:
            return float(os.getenv(key, str(default)))
        except (ValueError, TypeError):
            logger.warning(f"Invalid float value for {key}, using default: {default}")
            return default

    def should_use_enhanced_consciousness(self) -> bool:
        """Determine if enhanced consciousness should be used based on current state."""
        if self.fallback_mode_active:
            logger.debug("Enhanced consciousness disabled due to fallback mode")
            return False

        if self.consciousness_level == ConsciousnessLevel.BASIC:
            return False

        if self.consciousness_level == ConsciousnessLevel.FALLBACK:
            return False

        return self.consciousness_level == ConsciousnessLevel.ENHANCED

    def get_prompt_enhancements(self) -> Dict[str, Any]:
        """Get current prompt enhancement configuration."""
        if not self.should_use_enhanced_consciousness():
            return {
                "chain_of_thought": False,
                "few_shot": False,
                "constitutional_ai": False,
                "mood_influence": False
            }

        return {
            "chain_of_thought": self.chain_of_thought.enabled,
            "few_shot": self.few_shot.enabled,
            "constitutional_ai": self.constitutional_ai.enabled,
            "mood_influence": self.mood_influence.enabled
        }

    def enable_fallback_mode(self, reason: str) -> None:
        """Enable fallback mode for consciousness generation."""
        if not self.fallback_mode_active:
            logger.warning(f"Enabling consciousness fallback mode: {reason}")
            self.fallback_mode_active = True
            self.last_fallback_reason = reason

    def disable_fallback_mode(self) -> None:
        """Disable fallback mode and return to enhanced consciousness."""
        if self.fallback_mode_active:
            logger.info("Disabling consciousness fallback mode, returning to enhanced processing")
            self.fallback_mode_active = False
            self.last_fallback_reason = None

    def get_mood_weights(self) -> Dict[str, float]:
        """Get mood influence weights for MoodTransitionAnalyzer."""
        if not self.mood_influence.enabled:
            # Return equal weights when mood influence is disabled
            return {
                "global_state": 0.33,
                "recent_events": 0.33,
                "conversation_sentiment": 0.34
            }

        return {
            "global_state": self.mood_influence.global_state_weight,
            "recent_events": self.mood_influence.recent_events_weight,
            "conversation_sentiment": self.mood_influence.conversation_sentiment_weight
        }

    def get_transition_thresholds(self) -> Dict[str, float]:
        """Get mood transition detection thresholds."""
        return {
            "significant_shift": self.mood_influence.significant_shift_threshold,
            "conversation_impact": self.mood_influence.conversation_impact_threshold,
            "event_reaction": self.mood_influence.event_reaction_threshold
        }

    def validate_configuration(self) -> Dict[str, Any]:
        """Validate current configuration and return status."""
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": []
        }

        # Validate mood weights sum to approximately 1.0
        mood_weights = self.get_mood_weights()
        total_weight = sum(mood_weights.values())
        if abs(total_weight - 1.0) > 0.01:  # Allow small floating point differences
            validation_results["warnings"].append(
                f"Mood weights sum to {total_weight:.3f}, expected ~1.0"
            )

        # Validate performance thresholds
        if self.performance.max_consciousness_processing_ms < 100:
            validation_results["warnings"].append(
                "Consciousness processing timeout is very low (<100ms), may cause frequent fallbacks"
            )

        if self.performance.max_total_response_time_ms < 1000:
            validation_results["warnings"].append(
                "Total response time limit is very low (<1s), may impact quality"
            )

        # Validate Chain-of-Thought configuration
        if self.chain_of_thought.enabled and self.chain_of_thought.steps_count < 3:
            validation_results["warnings"].append(
                "Chain-of-Thought enabled but step count is low (<3)"
            )

        # Validate Few-Shot configuration
        if self.few_shot.enabled and self.few_shot.example_count == 0:
            validation_results["errors"].append(
                "Few-Shot learning enabled but example count is 0"
            )
            validation_results["valid"] = False

        return validation_results

    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration for logging/debugging."""
        return {
            "consciousness_level": self.consciousness_level.value,
            "fallback_mode_active": self.fallback_mode_active,
            "last_fallback_reason": self.last_fallback_reason,
            "enhancements": self.get_prompt_enhancements(),
            "mood_weights": self.get_mood_weights(),
            "performance_limits": {
                "consciousness_processing_ms": self.performance.max_consciousness_processing_ms,
                "total_response_ms": self.performance.max_total_response_time_ms,
                "mood_processing_ms": self.mood_influence.max_processing_time_ms
            },
            "validation": self.validate_configuration()
        }

    @classmethod
    def from_env(cls) -> 'ConsciousnessConfig':
        """Create consciousness configuration from environment variables."""
        try:
            config = cls()

            # Validate the configuration after creation
            validation = config.validate_configuration()
            if not validation["valid"]:
                logger.error(f"Invalid consciousness configuration: {validation['errors']}")
                # Fall back to basic configuration
                config.consciousness_level = ConsciousnessLevel.BASIC

            if validation["warnings"]:
                for warning in validation["warnings"]:
                    logger.warning(f"Consciousness config warning: {warning}")

            return config

        except Exception as e:
            logger.error(f"Failed to create consciousness configuration: {e}")
            # Return basic configuration as fallback
            fallback_config = cls()
            fallback_config.consciousness_level = ConsciousnessLevel.BASIC
            fallback_config.enable_fallback_mode(f"Configuration creation failed: {str(e)}")
            return fallback_config


# Global consciousness configuration instance
consciousness_config = ConsciousnessConfig.from_env()


def get_consciousness_config() -> ConsciousnessConfig:
    """Get global consciousness configuration instance."""
    return consciousness_config


def reload_consciousness_config() -> ConsciousnessConfig:
    """Reload consciousness configuration from environment variables."""
    global consciousness_config
    consciousness_config = ConsciousnessConfig.from_env()
    logger.info("Consciousness configuration reloaded")
    return consciousness_config