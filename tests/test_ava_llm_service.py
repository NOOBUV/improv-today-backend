"""
Tests for AvaLLMService
"""
import pytest
import json
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from dataclasses import dataclass

from app.services.ava_llm_service import AvaLLMService, AvaResponse
from app.services.conversation_prompt_service import EmotionType


@pytest.fixture
def llm_service():
    """Create AvaLLMService instance for testing"""
    return AvaLLMService()


@pytest.fixture
def mock_openai_response():
    """Create mock OpenAI response"""
    @dataclass
    class MockChoice:
        message: Mock
    
    @dataclass
    class MockMessage:
        content: str
    
    response = Mock()
    response.choices = [MockChoice(message=MockMessage(
        content='{"message": "Test response", "emotion": "calm"}'
    ))]
    return response


class TestAvaLLMService:
    """Test suite for AvaLLMService"""
    
    def test_init_without_api_key(self):
        """Test initialization without OpenAI API key"""
        with patch('app.services.ava_llm_service.settings') as mock_settings:
            mock_settings.openai_api_key = ""
            service = AvaLLMService()
            assert service.client is None
    
    def test_init_with_api_key(self):
        """Test initialization with OpenAI API key"""
        with patch('app.services.ava_llm_service.settings') as mock_settings:
            mock_settings.openai_api_key = "test-key"
            with patch('app.services.ava_llm_service.OpenAI') as mock_openai:
                service = AvaLLMService()
                mock_openai.assert_called_once_with(api_key="test-key")
    
    def test_init_with_api_error(self):
        """Test initialization with OpenAI API error"""
        with patch('app.services.ava_llm_service.settings') as mock_settings:
            mock_settings.openai_api_key = "test-key"
            with patch('app.services.ava_llm_service.OpenAI', side_effect=Exception("API Error")):
                service = AvaLLMService()
                assert service.client is None
    
    @pytest.mark.asyncio
    async def test_generate_ava_response_no_client(self, llm_service):
        """Test response generation without OpenAI client"""
        llm_service.client = None
        
        result = await llm_service.generate_ava_response("test prompt")
        
        assert isinstance(result, AvaResponse)
        assert not result.success
        assert result.emotion == EmotionType.CALM
        assert len(result.message) > 0
        assert "FALLBACK" in result.raw_response
    
    @pytest.mark.asyncio
    async def test_generate_ava_response_success(self, llm_service, mock_openai_response):
        """Test successful response generation"""
        llm_service.client = Mock()
        
        with patch.object(llm_service, '_make_openai_call', return_value=mock_openai_response):
            result = await llm_service.generate_ava_response("test prompt")
            
            assert isinstance(result, AvaResponse)
            assert result.success
            assert result.message == "Test response"
            assert result.emotion == EmotionType.CALM
    
    @pytest.mark.asyncio
    async def test_generate_ava_response_timeout(self, llm_service):
        """Test response generation with timeout"""
        llm_service.client = Mock()
        
        with patch.object(llm_service, '_make_openai_call', side_effect=asyncio.TimeoutError):
            result = await llm_service.generate_ava_response("test prompt", timeout=1)
            
            assert isinstance(result, AvaResponse)
            assert not result.success
            assert "timeout" in result.raw_response.lower()
    
    @pytest.mark.asyncio
    async def test_generate_ava_response_api_error(self, llm_service):
        """Test response generation with API error"""
        llm_service.client = Mock()
        
        with patch.object(llm_service, '_make_openai_call', side_effect=Exception("API Error")):
            result = await llm_service.generate_ava_response("test prompt")
            
            assert isinstance(result, AvaResponse)
            assert not result.success
            assert "API Error" in result.raw_response
    
    @pytest.mark.asyncio
    async def test_make_openai_call(self, llm_service):
        """Test OpenAI API call"""
        mock_client = Mock()
        mock_response = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        llm_service.client = mock_client
        
        result = await llm_service._make_openai_call("test prompt", 150, 0.8)
        
        assert result == mock_response
        mock_client.chat.completions.create.assert_called_once()
        
        # Check call arguments
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]["model"] == "gpt-4o"
        assert call_args[1]["max_tokens"] == 150
        assert call_args[1]["temperature"] == 0.8
        assert call_args[1]["response_format"] == {"type": "json_object"}
        
        # Check messages
        messages = call_args[1]["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "test prompt"
    
    def test_parse_ava_response_valid_json(self, llm_service, mock_openai_response):
        """Test parsing valid JSON response"""
        result = llm_service._parse_ava_response(mock_openai_response)
        
        assert isinstance(result, AvaResponse)
        assert result.success
        assert result.message == "Test response"
        assert result.emotion == EmotionType.CALM
    
    def test_parse_ava_response_invalid_emotion(self, llm_service):
        """Test parsing response with invalid emotion"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"message": "Test", "emotion": "invalid"}'
        
        result = llm_service._parse_ava_response(mock_response)
        
        assert result.emotion == EmotionType.CALM  # Should default to calm
        assert result.message == "Test"
    
    def test_parse_ava_response_missing_message(self, llm_service):
        """Test parsing response with missing message"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"emotion": "happy"}'
        
        result = llm_service._parse_ava_response(mock_response)
        
        assert not result.success
        assert result.emotion == EmotionType.CALM
    
    def test_parse_ava_response_invalid_json(self, llm_service):
        """Test parsing invalid JSON response"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = 'Invalid JSON content'
        
        result = llm_service._parse_ava_response(mock_response)
        
        assert not result.success
        assert result.message == "Invalid JSON content"
        assert result.emotion == EmotionType.CALM
    
    def test_get_fallback_response(self, llm_service):
        """Test fallback response generation"""
        result = llm_service._get_fallback_response()
        
        assert isinstance(result, AvaResponse)
        assert not result.success
        assert result.emotion == EmotionType.CALM
        assert len(result.message) > 0
        assert "FALLBACK" in result.raw_response
    
    def test_get_fallback_response_with_error(self, llm_service):
        """Test fallback response with error message"""
        result = llm_service._get_fallback_response("Test error")
        
        assert "Test error" in result.raw_response
    
    def test_validate_response_format_valid(self, llm_service):
        """Test response format validation with valid data"""
        valid_response = '{"message": "Test message", "emotion": "happy"}'
        
        is_valid, data = llm_service.validate_response_format(valid_response)
        
        assert is_valid
        assert data["message"] == "Test message"
        assert data["emotion"] == "happy"
    
    def test_validate_response_format_missing_message(self, llm_service):
        """Test response format validation with missing message"""
        invalid_response = '{"emotion": "happy"}'
        
        is_valid, data = llm_service.validate_response_format(invalid_response)
        
        assert not is_valid
        assert data is None
    
    def test_validate_response_format_missing_emotion(self, llm_service):
        """Test response format validation with missing emotion"""
        invalid_response = '{"message": "Test message"}'
        
        is_valid, data = llm_service.validate_response_format(invalid_response)
        
        assert not is_valid
        assert data is None
    
    def test_validate_response_format_invalid_emotion(self, llm_service):
        """Test response format validation with invalid emotion"""
        invalid_response = '{"message": "Test", "emotion": "invalid_emotion"}'
        
        is_valid, data = llm_service.validate_response_format(invalid_response)
        
        assert not is_valid
        assert data is None
    
    def test_validate_response_format_invalid_json(self, llm_service):
        """Test response format validation with invalid JSON"""
        invalid_response = 'Not valid JSON'
        
        is_valid, data = llm_service.validate_response_format(invalid_response)
        
        assert not is_valid
        assert data is None
    
    def test_fallback_responses_variety(self, llm_service):
        """Test that fallback responses provide variety"""
        responses = set()
        
        # Generate multiple fallback responses with different timings
        import time
        for i in range(20):
            # Small delay to ensure different timestamps
            time.sleep(0.001)
            result = llm_service._get_fallback_response()
            responses.add(result.message)
        
        # Should have more than one unique response
        assert len(responses) > 1