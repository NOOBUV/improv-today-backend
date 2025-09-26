"""
Tests for enhanced consciousness generator with 2025 prompt engineering techniques.
Compares outputs from original vs enhanced _build_consciousness_prompt() for same events.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch
from datetime import datetime

from app.services.consciousness_generator_service import ConsciousnessGeneratorService, ConsciousnessResponse
from app.models.simulation import GlobalEvents
from app.schemas.simulation_schemas import EventType, MoodImpact, ImpactLevel


@pytest.fixture
def mock_work_event():
    """Create a mock work event for testing."""
    event = Mock(spec=GlobalEvents)
    event.event_id = "test-work-123"
    event.event_type = EventType.WORK
    event.summary = "Important presentation went well, got positive feedback from manager"
    event.intensity = 6
    event.impact_mood = MoodImpact.POSITIVE
    event.impact_energy = ImpactLevel.STABLE
    event.impact_stress = ImpactLevel.DECREASE
    event.timestamp = datetime.now()
    return event


@pytest.fixture
def mock_social_event():
    """Create a mock social event for testing."""
    event = Mock(spec=GlobalEvents)
    event.event_id = "test-social-456"
    event.event_type = EventType.SOCIAL
    event.summary = "Friend invited me to a party but I'm feeling overwhelmed with work"
    event.intensity = 4
    event.impact_mood = MoodImpact.MIXED
    event.impact_energy = ImpactLevel.STABLE
    event.impact_stress = ImpactLevel.SLIGHT_INCREASE
    event.timestamp = datetime.now()
    return event


@pytest.fixture
def high_stress_context():
    """Create high stress state context for testing."""
    return {
        "mood": {"numeric_value": 35, "trend": "decreasing"},
        "energy": {"numeric_value": 40, "trend": "decreasing"},
        "stress": {"numeric_value": 85, "trend": "increasing"},
        "work_satisfaction": {"numeric_value": 30, "trend": "decreasing"},
        "social_satisfaction": {"numeric_value": 50, "trend": "stable"},
        "personal_fulfillment": {"numeric_value": 25, "trend": "decreasing"}
    }


@pytest.fixture
def balanced_context():
    """Create balanced state context for testing."""
    return {
        "mood": {"numeric_value": 65, "trend": "stable"},
        "energy": {"numeric_value": 70, "trend": "stable"},
        "stress": {"numeric_value": 45, "trend": "stable"},
        "work_satisfaction": {"numeric_value": 70, "trend": "stable"},
        "social_satisfaction": {"numeric_value": 75, "trend": "increasing"},
        "personal_fulfillment": {"numeric_value": 60, "trend": "stable"}
    }


@pytest.fixture
def consciousness_service():
    """Create ConsciousnessGeneratorService instance for testing."""
    return ConsciousnessGeneratorService()


class TestEnhancedConsciousnessPrompt:
    """Test cases for enhanced consciousness prompt with 2025 techniques."""

    @pytest.mark.asyncio
    async def test_enhanced_prompt_includes_chain_of_thought(self, consciousness_service, mock_work_event, balanced_context):
        """Test that enhanced prompt includes Chain-of-Thought reasoning structure."""

        with patch.object(consciousness_service.character_service, 'get_consolidated_backstory', return_value="Test backstory"):
            prompt = await consciousness_service._build_consciousness_prompt(mock_work_event, balanced_context)

        # Check for Chain-of-Thought elements
        assert "Chain-of-Thought Reasoning Process" in prompt
        assert "1. **Analyze the Event**" in prompt
        assert "2. **Consider Current State**" in prompt
        assert "3. **Reflect on Personality**" in prompt
        assert "4. **Determine Emotional Impact**" in prompt
        assert "5. **Choose Authentic Action**" in prompt
        assert "6. **Generate Internal Thoughts**" in prompt

    @pytest.mark.asyncio
    async def test_enhanced_prompt_includes_few_shot_examples(self, consciousness_service, mock_social_event, high_stress_context):
        """Test that enhanced prompt includes Few-Shot learning examples."""

        with patch.object(consciousness_service.character_service, 'get_consolidated_backstory', return_value="Test backstory"):
            prompt = await consciousness_service._build_consciousness_prompt(mock_social_event, high_stress_context)

        # Check for Few-Shot examples
        assert "Few-Shot Examples of Authentic Clara Responses" in prompt
        assert "**Example 1 - Work Event:**" in prompt
        assert "**Example 2 - Social Event:**" in prompt
        assert "**Example 3 - Personal Event:**" in prompt
        assert "reasoning_steps" in prompt  # Should be in examples
        assert "Client meeting went really well" in prompt  # Example content

    @pytest.mark.asyncio
    async def test_enhanced_prompt_includes_constitutional_ai_principles(self, consciousness_service, mock_work_event, balanced_context):
        """Test that enhanced prompt includes Constitutional AI principles."""

        with patch.object(consciousness_service.character_service, 'get_consolidated_backstory', return_value="Test backstory"):
            prompt = await consciousness_service._build_consciousness_prompt(mock_work_event, balanced_context)

        # Check for Constitutional AI principles
        assert "Constitutional AI Principles for Character Authenticity" in prompt
        assert "**Authenticity Over Generic AI**" in prompt
        assert "**Emotional Depth**" in prompt
        assert "**Character Growth**" in prompt
        assert "**Realistic Responses**" in prompt
        assert "**Human Vulnerability**" in prompt

    @pytest.mark.asyncio
    async def test_enhanced_prompt_requests_reasoning_steps(self, consciousness_service, mock_work_event, balanced_context):
        """Test that enhanced prompt requests reasoning_steps in JSON response."""

        with patch.object(consciousness_service.character_service, 'get_consolidated_backstory', return_value="Test backstory"):
            prompt = await consciousness_service._build_consciousness_prompt(mock_work_event, balanced_context)

        # Check for reasoning_steps in JSON format
        assert '"reasoning_steps": "Walk through steps 1-6' in prompt
        assert "Chain-of-Thought process" in prompt

    @pytest.mark.asyncio
    async def test_enhanced_prompt_uses_clara_name_consistently(self, consciousness_service, mock_social_event, high_stress_context):
        """Test that enhanced prompt uses Clara name consistently instead of Ava."""

        with patch.object(consciousness_service.character_service, 'get_consolidated_backstory', return_value="Test backstory"):
            prompt = await consciousness_service._build_consciousness_prompt(mock_social_event, high_stress_context)

        # Check character name consistency
        assert "You are Clara" in prompt
        assert "Clara's character" in prompt
        assert "Ava" not in prompt  # Should not use old character name


class TestEnhancedConsciousnessResponse:
    """Test cases for enhanced consciousness response parsing."""

    @pytest.mark.asyncio
    async def test_enhanced_response_parsing_with_reasoning_steps(self, consciousness_service, mock_work_event):
        """Test parsing enhanced response with reasoning_steps field."""

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """{
            "reasoning_steps": "1. This presentation success validates my hard work and skills. 2. My current balanced mood and moderate stress make me receptive to positive feedback. 3. I typically get motivated by professional recognition but also worry about maintaining standards. 4. I feel proud and energized. 5. I want to build on this momentum while staying grounded. 6. I'm thinking about how to leverage this success for future opportunities.",
            "emotional_reaction": "I'm genuinely thrilled and feel a real sense of accomplishment from this positive feedback. There's this warm validation that my hard work is being recognized and valued.",
            "chosen_action": "I'm going to take a moment to really savor this success, then use this momentum to tackle my next project with confidence.",
            "internal_thoughts": "This feels so good! I've been working so hard lately and it's amazing to see it paying off. I should celebrate this but also think about how to keep building on these skills."
        }"""

        result = consciousness_service._parse_consciousness_response(mock_response, mock_work_event)

        assert result.success is True
        assert "thrilled and feel a real sense" in result.emotional_reaction
        assert "savor this success" in result.chosen_action
        assert "This feels so good" in result.internal_thoughts
        # Note: reasoning_steps is logged but not stored in ConsciousnessResponse object

    @pytest.mark.asyncio
    async def test_enhanced_response_parsing_backward_compatibility(self, consciousness_service, mock_work_event):
        """Test that enhanced parsing works with old response format (without reasoning_steps)."""

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """{
            "emotional_reaction": "I'm really happy about how the presentation went and feel proud of my preparation.",
            "chosen_action": "I'll build on this confidence for my next big work challenge.",
            "internal_thoughts": "It's nice when preparation and effort pay off like this."
        }"""

        result = consciousness_service._parse_consciousness_response(mock_response, mock_work_event)

        assert result.success is True
        assert "really happy" in result.emotional_reaction
        assert "build on this confidence" in result.chosen_action
        assert "preparation and effort" in result.internal_thoughts

    def test_enhanced_response_format_validation_with_reasoning_steps(self, consciousness_service):
        """Test response format validation accepts reasoning_steps as optional."""

        enhanced_json = """{
            "reasoning_steps": "Step-by-step analysis of the situation and emotional processing.",
            "emotional_reaction": "I feel excited and nervous about this opportunity.",
            "chosen_action": "I'll prepare thoroughly while managing my expectations.",
            "internal_thoughts": "This could be really good for me if I handle it right."
        }"""

        is_valid, data = consciousness_service.validate_response_format(enhanced_json)
        assert is_valid is True
        assert "reasoning_steps" in data
        assert data["reasoning_steps"] == "Step-by-step analysis of the situation and emotional processing."

    def test_enhanced_response_format_validation_without_reasoning_steps(self, consciousness_service):
        """Test response format validation still works without reasoning_steps."""

        standard_json = """{
            "emotional_reaction": "I feel motivated by this challenge.",
            "chosen_action": "I'll approach this systematically.",
            "internal_thoughts": "I can handle this if I stay organized."
        }"""

        is_valid, data = consciousness_service.validate_response_format(standard_json)
        assert is_valid is True
        assert data["emotional_reaction"] == "I feel motivated by this challenge."


class TestEnhancedPromptComparison:
    """Test cases comparing original vs enhanced prompt outputs for same events."""

    @pytest.mark.asyncio
    async def test_enhanced_prompt_longer_than_original(self, consciousness_service, mock_work_event, balanced_context):
        """Test that enhanced prompt is significantly longer due to additional techniques."""

        # Mock the original simplified prompt (simulation of what it might have been)
        original_prompt_length = 800  # Approximate original length

        with patch.object(consciousness_service.character_service, 'get_consolidated_backstory', return_value="Test backstory"):
            enhanced_prompt = await consciousness_service._build_consciousness_prompt(mock_work_event, balanced_context)

        # Enhanced prompt should be significantly longer due to CoT, Few-Shot, and Constitutional AI
        assert len(enhanced_prompt) > original_prompt_length * 3  # At least 3x longer

    @pytest.mark.asyncio
    async def test_enhanced_prompt_contains_structured_thinking(self, consciousness_service, mock_social_event, high_stress_context):
        """Test that enhanced prompt provides structured thinking framework."""

        with patch.object(consciousness_service.character_service, 'get_consolidated_backstory', return_value="Test backstory"):
            prompt = await consciousness_service._build_consciousness_prompt(mock_social_event, high_stress_context)

        # Should contain structured elements that guide better reasoning
        structured_elements = [
            "step by step",
            "Think through",
            "Constitutional AI",
            "Examples of Authentic",
            "Character Authenticity"
        ]

        for element in structured_elements:
            assert element in prompt, f"Enhanced prompt missing structured element: {element}"

    @pytest.mark.asyncio
    async def test_enhanced_prompt_specific_to_context(self, consciousness_service, mock_work_event, high_stress_context):
        """Test that enhanced prompt includes context-specific guidance."""

        with patch.object(consciousness_service.character_service, 'get_consolidated_backstory', return_value="Test backstory"):
            prompt = await consciousness_service._build_consciousness_prompt(mock_work_event, high_stress_context)

        # Should include the specific event and state context
        assert "Important presentation went well" in prompt  # Event summary
        assert "Mood: 35/100" in prompt  # High stress context
        assert "Stress: 85/100" in prompt  # High stress context

        # Should provide contextually relevant examples and guidance
        assert "Work Event" in prompt  # Relevant example type
        assert "emotional nuance" in prompt  # Enhanced guidance


class TestEnhancedCharacterConsistency:
    """Test cases for enhanced character consistency validation."""

    def test_enhanced_character_validation_detects_ai_language(self, consciousness_service):
        """Test that character consistency validation catches AI-breaking language."""

        ai_response = {
            "reasoning_steps": "As an AI, I will analyze this situation logically.",
            "emotional_reaction": "I don't have feelings, but I can help you understand emotions.",
            "chosen_action": "I'm programmed to provide helpful responses.",
            "internal_thoughts": "My training data suggests this is a common scenario."
        }

        is_valid = consciousness_service._validate_character_consistency(ai_response)
        assert is_valid is False

    def test_enhanced_character_validation_accepts_human_response(self, consciousness_service):
        """Test that character consistency validation accepts genuinely human response."""

        human_response = {
            "reasoning_steps": "This reminds me of when I was in college and felt similar pressure about presentations.",
            "emotional_reaction": "I feel this mix of pride and anxiety - proud that it went well but worried about the next challenge.",
            "chosen_action": "I'm going to call my mom to share the good news, then maybe treat myself to something nice.",
            "internal_thoughts": "Why do I always worry about the next thing even when something goes well? I should just enjoy this moment."
        }

        is_valid = consciousness_service._validate_character_consistency(human_response)
        assert is_valid is True

    def test_enhanced_character_validation_handles_reasoning_steps(self, consciousness_service):
        """Test that character consistency validation works with new reasoning_steps field."""

        response_with_reasoning = {
            "reasoning_steps": "I need to think through how this affects my relationships and my own wellbeing.",
            "emotional_reaction": "I'm feeling conflicted about whether to prioritize my friend's needs or my own boundaries.",
            "chosen_action": "I'll have an honest conversation about what I can realistically commit to right now.",
            "internal_thoughts": "It's hard to disappoint people, but I'm learning that saying no sometimes is necessary for my mental health."
        }

        is_valid = consciousness_service._validate_character_consistency(response_with_reasoning)
        assert is_valid is True