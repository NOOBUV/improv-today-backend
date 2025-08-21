import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from app.services.simple_openai import SimpleOpenAIService, OpenAICoachingResponse, WordUsageStatus


class TestOpenAICoaching:
    """Test OpenAI coaching response functionality and edge cases."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.openai_service = SimpleOpenAIService()
        self.test_message = "Hello, I want to elaborate on this topic"
        self.conversation_history = "User: Hi there\nAssistant: Hello! How are you today?"
        self.suggested_word = "elaborate"
    
    @pytest.mark.asyncio
    @patch('app.services.simple_openai.settings.openai_api_key', 'test-key')
    async def test_coaching_response_word_used_correctly(self):
        """Test coaching response when suggested word is used correctly."""
        mock_response = {
            "corrected_transcript": "Hello, I want to elaborate on this topic.",
            "ai_response": "That's great! Please elaborate further on what interests you most.",
            "word_usage_status": "used_correctly", 
            "usage_correctness_feedback": None
        }
        
        with patch.object(self.openai_service, 'client') as mock_client:
            mock_completion = Mock()
            mock_completion.choices = [Mock()]
            mock_completion.choices[0].message.content = json.dumps(mock_response)
            mock_client.chat.completions.create.return_value = mock_completion
            
            result = await self.openai_service.generate_coaching_response(
                self.test_message,
                self.conversation_history,
                "friendly_neutral",
                [],
                self.suggested_word
            )
            
            assert isinstance(result, OpenAICoachingResponse)
            assert result.word_usage_status == WordUsageStatus.USED_CORRECTLY
            assert result.usage_correctness_feedback is None
            assert "elaborate" in result.corrected_transcript
            assert result.ai_response == mock_response["ai_response"]
    
    @pytest.mark.asyncio
    @patch('app.services.simple_openai.settings.openai_api_key', 'test-key')
    async def test_coaching_response_word_used_incorrectly(self):
        """Test coaching response when suggested word is used incorrectly."""
        mock_response = {
            "corrected_transcript": "Hello, I want to elaborate about this topic.",
            "ai_response": "I understand you want to expand on that topic. Can you tell me more?",
            "word_usage_status": "used_incorrectly",
            "usage_correctness_feedback": "The word 'elaborate' should be followed by 'on' not 'about' in this context."
        }
        
        with patch.object(self.openai_service, 'client') as mock_client:
            mock_completion = Mock()
            mock_completion.choices = [Mock()]
            mock_completion.choices[0].message.content = json.dumps(mock_response)
            mock_client.chat.completions.create.return_value = mock_completion
            
            result = await self.openai_service.generate_coaching_response(
                "Hello, I want to elaborate about this topic",
                self.conversation_history,
                "friendly_neutral",
                [],
                self.suggested_word
            )
            
            assert result.word_usage_status == WordUsageStatus.USED_INCORRECTLY
            assert result.usage_correctness_feedback is not None
            assert "elaborate" in result.usage_correctness_feedback
    
    @pytest.mark.asyncio
    @patch('app.services.simple_openai.settings.openai_api_key', 'test-key')
    async def test_coaching_response_word_not_used(self):
        """Test coaching response when suggested word is not used."""
        mock_response = {
            "corrected_transcript": "Hello, I want to discuss this topic further.",
            "ai_response": "That sounds interesting! What aspects would you like to explore?",
            "word_usage_status": "not_used",
            "usage_correctness_feedback": None
        }
        
        with patch.object(self.openai_service, 'client') as mock_client:
            mock_completion = Mock()
            mock_completion.choices = [Mock()]
            mock_completion.choices[0].message.content = json.dumps(mock_response)
            mock_client.chat.completions.create.return_value = mock_completion
            
            result = await self.openai_service.generate_coaching_response(
                "Hello, I want to discuss this topic further",
                self.conversation_history,
                "friendly_neutral",
                [],
                self.suggested_word
            )
            
            assert result.word_usage_status == WordUsageStatus.NOT_USED
            assert result.usage_correctness_feedback is None
    
    @pytest.mark.asyncio
    @patch('app.services.simple_openai.settings.openai_api_key', '')
    async def test_coaching_response_no_api_key_fallback(self):
        """Test coaching response fallback when no API key is available."""
        result = await self.openai_service.generate_coaching_response(
            self.test_message,
            self.conversation_history,
            "friendly_neutral",
            [],
            self.suggested_word
        )
        
        assert isinstance(result, OpenAICoachingResponse)
        assert result.corrected_transcript == self.test_message
        assert result.word_usage_status == WordUsageStatus.NOT_USED
        assert result.usage_correctness_feedback is None
        # Should contain fallback response
        assert len(result.ai_response) > 0
    
    @pytest.mark.asyncio
    @patch('app.services.simple_openai.settings.openai_api_key', 'test-key')
    async def test_coaching_response_json_parsing_error(self):
        """Test coaching response when JSON parsing fails."""
        with patch.object(self.openai_service, 'client') as mock_client:
            mock_completion = Mock()
            mock_completion.choices = [Mock()]
            mock_completion.choices[0].message.content = "Invalid JSON response"
            mock_client.chat.completions.create.return_value = mock_completion
            
            # Mock the fallback method
            with patch.object(self.openai_service, '_handle_coaching_response_fallback', 
                            return_value=OpenAICoachingResponse(
                                corrected_transcript=self.test_message,
                                ai_response="Fallback response",
                                word_usage_status=WordUsageStatus.NOT_USED,
                                usage_correctness_feedback=None
                            )) as mock_fallback:
                
                result = await self.openai_service.generate_coaching_response(
                    self.test_message,
                    self.conversation_history,
                    "friendly_neutral",
                    [],
                    self.suggested_word
                )
                
                # Verify fallback was called
                mock_fallback.assert_called_once()
                assert result.ai_response == "Fallback response"
    
    @pytest.mark.asyncio
    @patch('app.services.simple_openai.settings.openai_api_key', 'test-key')
    async def test_coaching_response_with_conversation_history(self):
        """Test that conversation history is included in the prompt."""
        mock_response = {
            "corrected_transcript": "Yes, I love reading mystery novels.",
            "ai_response": "Given our earlier discussion about books, what's your favorite mystery author?",
            "word_usage_status": "not_used",
            "usage_correctness_feedback": None
        }
        
        conversation_history = "User: I enjoy reading books\nAssistant: What genre do you prefer?"
        
        with patch.object(self.openai_service, 'client') as mock_client:
            mock_completion = Mock()
            mock_completion.choices = [Mock()]
            mock_completion.choices[0].message.content = json.dumps(mock_response)
            mock_client.chat.completions.create.return_value = mock_completion
            
            result = await self.openai_service.generate_coaching_response(
                "Yes, I love reading mystery novels",
                conversation_history,
                "friendly_neutral",
                [],
                None
            )
            
            # Verify the call was made with conversation history
            call_args = mock_client.chat.completions.create.call_args
            system_prompt = call_args[1]['messages'][0]['content']
            
            # Check that conversation history was included in prompt
            assert "Recent conversation history:" in system_prompt
            assert "I enjoy reading books" in system_prompt
            assert "What genre do you prefer?" in system_prompt
    
    def test_word_usage_status_enum_values(self):
        """Test WordUsageStatus enum has correct values."""
        assert WordUsageStatus.NOT_USED == "not_used"
        assert WordUsageStatus.USED_CORRECTLY == "used_correctly"
        assert WordUsageStatus.USED_INCORRECTLY == "used_incorrectly"
    
    @pytest.mark.asyncio
    @patch('app.services.simple_openai.settings.openai_api_key', 'test-key')
    async def test_coaching_response_empty_conversation_history(self):
        """Test coaching response with empty conversation history."""
        mock_response = {
            "corrected_transcript": "Hello there!",
            "ai_response": "Hi! How can I help you today?",
            "word_usage_status": "not_used",
            "usage_correctness_feedback": None
        }
        
        with patch.object(self.openai_service, 'client') as mock_client:
            mock_completion = Mock()
            mock_completion.choices = [Mock()]
            mock_completion.choices[0].message.content = json.dumps(mock_response)
            mock_client.chat.completions.create.return_value = mock_completion
            
            result = await self.openai_service.generate_coaching_response(
                "Hello there!",
                "",  # Empty conversation history
                "friendly_neutral",
                [],
                None
            )
            
            # Should still work with empty history
            assert isinstance(result, OpenAICoachingResponse)
            assert result.corrected_transcript == "Hello there!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])