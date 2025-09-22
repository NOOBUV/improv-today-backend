import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from app.services.simple_openai import SimpleOpenAIService, OpenAICoachingResponse, WordUsageStatus
from app.services.redis_service import RedisService


class TestStory21Integration:
    """Integration tests for Story 2.1: Context-Aware AI Prompts functionality."""
    
    @pytest.mark.asyncio
    @patch('app.services.simple_openai.settings.openai_api_key', 'test-key')
    async def test_end_to_end_coaching_with_conversation_history(self):
        """Test complete coaching workflow with conversation history integration."""
        
        # Setup services
        openai_service = SimpleOpenAIService()
        redis_service = RedisService()
        
        # Mock conversation history data
        conversation_history_data = [
            {"role": "user", "content": "I love reading books", "timestamp": "2023-01-01T00:00:00"},
            {"role": "assistant", "content": "That's wonderful! What genres do you enjoy?", "timestamp": "2023-01-01T00:01:00"},
            {"role": "user", "content": "I prefer mystery novels", "timestamp": "2023-01-01T00:02:00"}
        ]
        
        # Mock OpenAI response for coaching
        mock_coaching_response = {
            "corrected_transcript": "I want to elaborate on my favorite mystery authors.",
            "ai_response": "Given our discussion about mystery novels, I'd love to hear more about your favorite authors! Who would you elaborate on first?",
            "word_usage_status": "used_correctly",
            "usage_correctness_feedback": None
        }
        
        with patch.object(openai_service, 'client') as mock_client:
            mock_completion = Mock()
            mock_completion.choices = [Mock()]
            mock_completion.choices[0].message.content = json.dumps(mock_coaching_response)
            mock_client.chat.completions.create.return_value = mock_completion
            
            # Build conversation context
            conversation_context = redis_service.build_conversation_context(conversation_history_data)
            
            # Test coaching response with conversation history and word usage evaluation
            result = await openai_service.generate_coaching_response(
                message="I want to elaborate on my favorite mystery authors",
                conversation_history=conversation_context,
                personality="friendly_neutral",
                target_vocabulary=[],
                suggested_word="elaborate"
            )
            
            # Verify AC: 3 - Structured JSON response with word usage analysis
            assert isinstance(result, OpenAICoachingResponse)
            assert result.word_usage_status == WordUsageStatus.USED_CORRECTLY
            assert result.usage_correctness_feedback is None
            assert "elaborate" in result.corrected_transcript
            
            # Verify AC: 2 - Conversation history was included in prompt
            call_args = mock_client.chat.completions.create.call_args
            system_prompt = call_args[1]['messages'][0]['content']
            assert "Recent conversation history:" in system_prompt
            assert "reading books" in system_prompt
            assert "mystery novels" in system_prompt
            
            # Verify AI response builds on conversation context
            assert "mystery novels" in result.ai_response or "authors" in result.ai_response
            
    def test_redis_service_conversation_history_caching(self):
        """Test Redis conversation history caching and retrieval."""
        redis_service = RedisService()
        conversation_id = "test-conversation-123"
        
        # Mock successful Redis operations
        with patch.object(redis_service, '_get_client') as mock_get_client:
            mock_client = Mock()
            mock_client.lpush.return_value = 1
            mock_client.expire.return_value = True
            mock_client.ltrim.return_value = True
            mock_client.lrange.return_value = [
                '{"role": "assistant", "content": "Great question!", "timestamp": "2023-01-01T00:01:00"}',
                '{"role": "user", "content": "How are you?", "timestamp": "2023-01-01T00:00:00"}'
            ]
            mock_get_client.return_value = mock_client
            
            # Test message caching
            cache_result = redis_service.cache_message(
                conversation_id, 
                "user", 
                "How are you?"
            )
            assert cache_result is True
            
            # Test conversation history retrieval
            mock_db = Mock()  # Won't be used since Redis is available
            history = redis_service.get_conversation_history(conversation_id, mock_db)
            
            # Verify AC: 1 - Retrieved last ~15 messages (we have 2)
            assert len(history) == 2
            assert history[0]["role"] == "user"
            assert history[0]["content"] == "How are you?"
            assert history[1]["role"] == "assistant"
            assert history[1]["content"] == "Great question!"
    
    def test_redis_fallback_to_database(self):
        """Test IV1: Redis fallback to database functionality."""
        redis_service = RedisService()
        conversation_id = "test-conversation-fallback"
        
        # Mock Redis unavailable
        with patch.object(redis_service, '_get_client', return_value=None):
            # Mock database session
            mock_db = Mock()
            from datetime import datetime, timezone
            
            mock_messages = [
                Mock(role="user", content="Hello there", timestamp=datetime.now(timezone.utc)),
                Mock(role="assistant", content="Hi! How are you?", timestamp=datetime.now(timezone.utc))
            ]
            mock_db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = mock_messages
            
            # Test fallback to database
            history = redis_service.get_conversation_history(conversation_id, mock_db)
            
            # Verify fallback worked
            assert len(history) == 2
            assert history[0]["role"] == "assistant"  # Reversed order
            assert history[1]["role"] == "user"
            
            # Verify database was queried
            mock_db.query.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_word_usage_status_evaluation_scenarios(self):
        """Test all WordUsageStatus enum scenarios."""
        
        test_scenarios = [
            {
                "status": WordUsageStatus.USED_CORRECTLY,
                "feedback": None,
                "description": "Word used correctly"
            },
            {
                "status": WordUsageStatus.USED_INCORRECTLY, 
                "feedback": "The word should be used in a different context",
                "description": "Word used incorrectly with feedback"
            },
            {
                "status": WordUsageStatus.NOT_USED,
                "feedback": None,
                "description": "Word not used at all"
            }
        ]
        
        for scenario in test_scenarios:
            # Create mock response for each scenario
            response = OpenAICoachingResponse(
                corrected_transcript="Test transcript",
                ai_response="Test AI response",
                word_usage_status=scenario["status"],
                usage_correctness_feedback=scenario["feedback"]
            )
            
            # Verify AC: 3 - Proper status and feedback handling
            assert response.word_usage_status == scenario["status"]
            assert response.usage_correctness_feedback == scenario["feedback"]
            
            # Verify feedback is only provided when status is USED_INCORRECTLY
            if scenario["status"] == WordUsageStatus.USED_INCORRECTLY:
                assert response.usage_correctness_feedback is not None
            else:
                assert response.usage_correctness_feedback is None
    
    def test_conversation_context_building(self):
        """Test conversation context building for OpenAI prompts."""
        redis_service = RedisService()
        
        # Test with realistic conversation history
        history_data = [
            {"role": "user", "content": "I'm learning to cook", "timestamp": "2023-01-01T00:00:00"},
            {"role": "assistant", "content": "That's exciting! What cuisine interests you?", "timestamp": "2023-01-01T00:01:00"},
            {"role": "user", "content": "I want to master Italian cuisine", "timestamp": "2023-01-01T00:02:00"},
            {"role": "assistant", "content": "Italian food is wonderful! Have you tried making pasta?", "timestamp": "2023-01-01T00:03:00"}
        ]
        
        context = redis_service.build_conversation_context(history_data)
        
        # Verify AC: 2 - Context includes conversation history for prompts
        expected_context = (
            "User: I'm learning to cook\n"
            "Assistant: That's exciting! What cuisine interests you?\n"
            "User: I want to master Italian cuisine\n"
            "Assistant: Italian food is wonderful! Have you tried making pasta?"
        )
        
        assert context == expected_context
        
        # Verify context can be used in prompts
        assert len(context) > 0
        assert "User:" in context
        assert "Assistant:" in context
        assert context.count("\n") == 3  # 4 messages = 3 newlines


if __name__ == "__main__":
    pytest.main([__file__, "-v"])