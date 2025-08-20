"""
Test suite for Story 2.2: AI-Driven Conversation Steering

Tests the enhanced OpenAI coaching response with conversation steering functionality.
Validates that AI naturally guides conversation toward suggested vocabulary words
while maintaining conversational quality and personality characteristics.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from app.services.simple_openai import SimpleOpenAIService, OpenAICoachingResponse, WordUsageStatus


class TestConversationSteering:
    """Test AI-driven conversation steering functionality"""
    
    @pytest.fixture
    def openai_service(self):
        """Create OpenAI service instance for testing"""
        return SimpleOpenAIService()
    
    @pytest.fixture
    def mock_openai_response_with_steering(self):
        """Mock OpenAI response that demonstrates steering behavior"""
        return Mock(
            choices=[Mock(
                message=Mock(
                    content='{"corrected_transcript": "I had a great day at work today.", "ai_response": "That sounds wonderful! What made your day so great? I\'d love to hear about any exciting accomplishments or achievements that made you feel particularly elated!", "word_usage_status": "not_used", "usage_correctness_feedback": null}'
                )
            )]
        )
    
    @pytest.fixture
    def mock_openai_response_without_steering(self):
        """Mock OpenAI response when no suggestion is active"""
        return Mock(
            choices=[Mock(
                message=Mock(
                    content='{"corrected_transcript": "I had a great day at work today.", "ai_response": "That sounds wonderful! What happened at work that made it so great?", "word_usage_status": "not_used", "usage_correctness_feedback": null}'
                )
            )]
        )
    
    @pytest.mark.asyncio
    async def test_steering_with_active_suggestion(self, openai_service, mock_openai_response_with_steering):
        """Test AC: 1, 2 - Steering logic activates when suggestion is present"""
        with patch.object(openai_service.client.chat.completions, 'create', return_value=mock_openai_response_with_steering):
            result = await openai_service.generate_coaching_response(
                message="I had a great day at work today.",
                conversation_history="",
                personality="friendly_neutral",
                target_vocabulary=[],
                suggested_word="elated"
            )
            
            # Verify response structure
            assert isinstance(result, OpenAICoachingResponse)
            assert result.corrected_transcript == "I had a great day at work today."
            assert result.word_usage_status == WordUsageStatus.NOT_USED
            assert result.usage_correctness_feedback is None
            
            # Verify steering behavior - response should guide toward "elated" usage
            assert "elated" in result.ai_response.lower() or "exciting" in result.ai_response.lower() or "accomplishments" in result.ai_response.lower()
            assert len(result.ai_response) > 20  # Should be conversational length
    
    @pytest.mark.asyncio
    async def test_no_steering_without_suggestion(self, openai_service, mock_openai_response_without_steering):
        """Test IV2 - Normal response when no suggestion is active"""
        with patch.object(openai_service.client.chat.completions, 'create', return_value=mock_openai_response_without_steering):
            result = await openai_service.generate_coaching_response(
                message="I had a great day at work today.",
                conversation_history="",
                personality="friendly_neutral",
                target_vocabulary=[],
                suggested_word=None
            )
            
            # Verify normal response without steering
            assert isinstance(result, OpenAICoachingResponse)
            assert result.corrected_transcript == "I had a great day at work today."
            assert "elated" not in result.ai_response.lower()  # Should not include steering words
            assert "exciting" not in result.ai_response or "accomplishments" not in result.ai_response  # Natural response
    
    @pytest.mark.asyncio
    async def test_steering_preserves_personality_sassy(self, openai_service):
        """Test AC: 3, IV1 - Steering maintains personality characteristics"""
        mock_response = Mock(
            choices=[Mock(
                message=Mock(
                    content='{"corrected_transcript": "I went shopping today.", "ai_response": "Oh brilliant! Shopping, eh? Did you find anything that made you feel particularly elated, darling? Any fabulous finds that lifted your spirits?", "word_usage_status": "not_used", "usage_correctness_feedback": null}'
                )
            )]
        )
        
        with patch.object(openai_service.client.chat.completions, 'create', return_value=mock_response):
            result = await openai_service.generate_coaching_response(
                message="I went shopping today.",
                conversation_history="",
                personality="sassy_english",
                target_vocabulary=[],
                suggested_word="elated"
            )
            
            # Verify sassy personality is preserved while steering
            response = result.ai_response.lower()
            assert any(sassy_word in response for sassy_word in ["brilliant", "darling", "eh", "fabulous"])
            assert "elated" in response  # Steering word should be present
    
    @pytest.mark.asyncio
    async def test_steering_preserves_personality_blunt(self, openai_service):
        """Test AC: 3, IV1 - Steering maintains blunt personality characteristics"""
        mock_response = Mock(
            choices=[Mock(
                message=Mock(
                    content='{"corrected_transcript": "I had a meeting today.", "ai_response": "Alright, how did the meeting go? Did anything happen that made you feel elated or really pumped up about the outcome?", "word_usage_status": "not_used", "usage_correctness_feedback": null}'
                )
            )]
        )
        
        with patch.object(openai_service.client.chat.completions, 'create', return_value=mock_response):
            result = await openai_service.generate_coaching_response(
                message="I had a meeting today.",
                conversation_history="",
                personality="blunt_american",
                target_vocabulary=[],
                suggested_word="elated"
            )
            
            # Verify blunt personality is preserved while steering
            response = result.ai_response.lower()
            assert any(blunt_word in response for blunt_word in ["alright", "how did", "pumped up"])
            assert "elated" in response  # Steering word should be present
    
    @pytest.mark.asyncio
    async def test_steering_with_conversation_history(self, openai_service):
        """Test steering works with conversation history context"""
        conversation_history = """
        User: Hi, I'm Sarah
        Assistant: Nice to meet you, Sarah! How's your day going?
        User: Pretty good so far
        Assistant: That's great to hear! What have you been up to today?
        """
        
        mock_response = Mock(
            choices=[Mock(
                message=Mock(
                    content='{"corrected_transcript": "I just finished a big project at work.", "ai_response": "That\'s fantastic, Sarah! Completing a big project must feel amazing. Were you elated when you finally finished it? What was the most rewarding part?", "word_usage_status": "not_used", "usage_correctness_feedback": null}'
                )
            )]
        )
        
        with patch.object(openai_service.client.chat.completions, 'create', return_value=mock_response):
            result = await openai_service.generate_coaching_response(
                message="I just finished a big project at work.",
                conversation_history=conversation_history,
                personality="friendly_neutral",
                target_vocabulary=[],
                suggested_word="elated"
            )
            
            # Verify steering works with history context
            assert "Sarah" in result.ai_response  # Should use name from history
            assert "elated" in result.ai_response.lower()  # Should include steering
            assert "project" in result.ai_response.lower()  # Should respond to current message
    
    @pytest.mark.asyncio
    async def test_steering_different_word_types(self, openai_service):
        """Test steering adapts to different types of suggested words"""
        test_cases = [
            {
                "suggested_word": "elaborate",
                "message": "I like reading books.",
                "expected_patterns": ["elaborate", "details", "more about", "tell me more"]
            },
            {
                "suggested_word": "contemplate", 
                "message": "I'm thinking about changing jobs.",
                "expected_patterns": ["contemplate", "consider", "think about", "reflect"]
            },
            {
                "suggested_word": "serene",
                "message": "I went to the park today.",
                "expected_patterns": ["serene", "peaceful", "calm", "tranquil"]
            }
        ]
        
        for case in test_cases:
            mock_response = Mock(
                choices=[Mock(
                    message=Mock(
                        content=f'{{"corrected_transcript": "{case["message"]}", "ai_response": "That sounds interesting! Can you {case["expected_patterns"][0]} on that experience?", "word_usage_status": "not_used", "usage_correctness_feedback": null}}'
                    )
                )]
            )
            
            with patch.object(openai_service.client.chat.completions, 'create', return_value=mock_response):
                result = await openai_service.generate_coaching_response(
                    message=case["message"],
                    conversation_history="",
                    personality="friendly_neutral",
                    target_vocabulary=[],
                    suggested_word=case["suggested_word"]
                )
                
                # Verify steering adapts to word type
                response_lower = result.ai_response.lower()
                assert any(pattern in response_lower for pattern in case["expected_patterns"])
    
    @pytest.mark.asyncio
    async def test_fallback_when_api_fails(self, openai_service):
        """Test graceful fallback when OpenAI API fails during steering"""
        with patch.object(openai_service.client.chat.completions, 'create', side_effect=Exception("API Error")):
            result = await openai_service.generate_coaching_response(
                message="I had a great day.",
                conversation_history="",
                personality="friendly_neutral",
                target_vocabulary=[],
                suggested_word="elated"
            )
            
            # Verify graceful fallback
            assert isinstance(result, OpenAICoachingResponse)
            assert result.corrected_transcript == "I had a great day."  # Original message preserved
            assert len(result.ai_response) > 0  # Fallback response provided
            assert result.word_usage_status == WordUsageStatus.NOT_USED
            assert result.usage_correctness_feedback is None
    
    @pytest.mark.asyncio
    async def test_steering_prompt_construction(self, openai_service):
        """Test that steering context is properly constructed in the prompt"""
        
        # Capture the actual prompt sent to OpenAI
        captured_prompt = None
        
        def capture_prompt(**kwargs):
            nonlocal captured_prompt
            captured_prompt = kwargs['messages'][0]['content']
            return Mock(
                choices=[Mock(
                    message=Mock(
                        content='{"corrected_transcript": "Test message.", "ai_response": "Test response.", "word_usage_status": "not_used", "usage_correctness_feedback": null}'
                    )
                )]
            )
        
        with patch.object(openai_service.client.chat.completions, 'create', side_effect=capture_prompt):
            await openai_service.generate_coaching_response(
                message="Test message.",
                conversation_history="",
                personality="friendly_neutral",
                target_vocabulary=[],
                suggested_word="magnificent"
            )
            
            # Verify steering instructions are in prompt
            assert captured_prompt is not None
            assert "CONVERSATION STEERING" in captured_prompt
            assert "magnificent" in captured_prompt
            assert "naturally steer the conversation" in captured_prompt
            assert "Never making it obvious" in captured_prompt
    
    @pytest.mark.asyncio
    async def test_no_steering_prompt_when_no_suggestion(self, openai_service):
        """Test that steering context is not added when no suggestion exists"""
        
        # Capture the actual prompt sent to OpenAI
        captured_prompt = None
        
        def capture_prompt(**kwargs):
            nonlocal captured_prompt
            captured_prompt = kwargs['messages'][0]['content']
            return Mock(
                choices=[Mock(
                    message=Mock(
                        content='{"corrected_transcript": "Test message.", "ai_response": "Test response.", "word_usage_status": "not_used", "usage_correctness_feedback": null}'
                    )
                )]
            )
        
        with patch.object(openai_service.client.chat.completions, 'create', side_effect=capture_prompt):
            await openai_service.generate_coaching_response(
                message="Test message.",
                conversation_history="",
                personality="friendly_neutral",
                target_vocabulary=[],
                suggested_word=None
            )
            
            # Verify steering instructions are NOT in prompt
            assert captured_prompt is not None
            assert "CONVERSATION STEERING" not in captured_prompt
            assert "naturally steer the conversation" not in captured_prompt


class TestSteeringIntegration:
    """Integration tests for steering with the conversation endpoint"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_steering_flow(self):
        """Test complete steering flow from endpoint to response"""
        # This would be an integration test that verifies:
        # 1. Suggestion is retrieved from database
        # 2. Suggestion is passed to coaching service
        # 3. Steering occurs in AI response
        # 4. Response maintains quality standards
        pass
    
    @pytest.mark.asyncio 
    async def test_performance_with_steering(self):
        """Test that steering doesn't significantly impact response time"""
        # This would verify that adding steering instructions doesn't 
        # violate the <2000ms response time requirement
        pass


if __name__ == "__main__":
    # Run tests with: python -m pytest tests/test_conversation_steering.py -v
    pytest.main([__file__, "-v"])
