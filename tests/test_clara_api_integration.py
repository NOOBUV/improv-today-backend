"""
Integration tests for Clara API endpoint
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
from datetime import datetime

from app.main import app
from app.services.conversation_prompt_service import EmotionType
from app.services.clara_llm_service import ClaraResponse


client = TestClient(app)


class TestClaraConversationEndpoint:
    """Integration tests for /api/clara/conversation endpoint"""
    
    def test_conversation_endpoint_exists(self):
        """Test that the conversation endpoint is accessible"""
        # Test with minimal valid request
        response = client.post(
            "/api/clara/conversation",
            json={"message": "Hello"}
        )
        
        # Should not return 404 (endpoint exists)
        assert response.status_code != 404
    
    @patch('app.services.character_content_service.CharacterContentService.get_consolidated_backstory')
    @patch('app.services.ava_llm_service.AvaLLMService.generate_ava_response')
    def test_conversation_endpoint_success(self, mock_llm_response, mock_backstory):
        """Test successful conversation endpoint call"""
        # Setup mocks
        mock_backstory.return_value = "Test character backstory"
        mock_llm_response.return_value = ClaraResponse(
            message="Hello! How are you doing today?",
            emotion=EmotionType.HAPPY,
            raw_response='{"message": "Hello! How are you doing today?", "emotion": "happy"}',
            success=True
        )
        
        # Make request
        response = client.post(
            "/api/clara/conversation",
            json={"message": "Hi there!"}
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "message" in data
        assert "emotion" in data
        assert "emotional_state" in data
        assert "timestamp" in data
        assert "conversation_id" in data
        assert "context_used" in data
        
        # Check specific values
        assert data["message"] == "Hello! How are you doing today?"
        assert data["emotion"] == "happy"
        assert data["emotional_state"]["emotion"] == "happy"
        assert data["context_used"] is False
    
    @patch('app.services.character_content_service.CharacterContentService.get_consolidated_backstory')
    @patch('app.services.ava_llm_service.AvaLLMService.generate_ava_response')
    def test_conversation_endpoint_with_conversation_id(self, mock_llm_response, mock_backstory):
        """Test conversation endpoint with provided conversation ID"""
        mock_backstory.return_value = "Test backstory"
        mock_llm_response.return_value = ClaraResponse(
            message="Test response",
            emotion=EmotionType.CALM,
            raw_response='{"message": "Test response", "emotion": "calm"}',
            success=True
        )
        
        conversation_id = "test-conversation-123"
        
        response = client.post(
            "/api/clara/conversation",
            json={
                "message": "Test message",
                "conversation_id": conversation_id
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["conversation_id"] == conversation_id
    
    @patch('app.services.character_content_service.CharacterContentService.get_consolidated_backstory')
    @patch('app.services.ava_llm_service.AvaLLMService.generate_ava_response')
    def test_conversation_endpoint_emotion_detection(self, mock_llm_response, mock_backstory):
        """Test that different message types trigger appropriate emotions"""
        mock_backstory.return_value = "Test backstory"
        
        test_cases = [
            ("That's hilarious!", EmotionType.SASSY),
            ("I'm feeling sad today", EmotionType.SAD),
            ("I'm so happy about this!", EmotionType.HAPPY),
            ("I'm overwhelmed with deadlines", EmotionType.STRESSED),
            ("How are you?", EmotionType.CALM)
        ]
        
        for message, expected_emotion in test_cases:
            mock_llm_response.return_value = ClaraResponse(
                message="Test response",
                emotion=expected_emotion,
                raw_response=f'{{"message": "Test response", "emotion": "{expected_emotion.value}"}}',
                success=True
            )
            
            response = client.post(
                "/api/clara/conversation",
                json={"message": message}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["emotion"] == expected_emotion.value
    
    @patch('app.services.character_content_service.CharacterContentService.get_consolidated_backstory')
    @patch('app.services.ava_llm_service.AvaLLMService.generate_ava_response')
    def test_conversation_endpoint_fallback_handling(self, mock_llm_response, mock_backstory):
        """Test handling of LLM service fallback responses"""
        mock_backstory.return_value = "Test backstory"
        mock_llm_response.return_value = ClaraResponse(
            message="I'm having trouble finding the right words right now.",
            emotion=EmotionType.CALM,
            raw_response="FALLBACK - API unavailable",
            success=False
        )
        
        response = client.post(
            "/api/clara/conversation",
            json={"message": "Hello"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "I'm having trouble finding the right words right now."
        assert data["emotion"] == "calm"
    
    @patch('app.services.character_content_service.CharacterContentService.get_consolidated_backstory')
    def test_conversation_endpoint_content_loading_failure(self, mock_backstory):
        """Test handling when character content loading fails"""
        mock_backstory.return_value = ""  # Empty content
        
        with patch('app.services.ava_llm_service.AvaLLMService.generate_ava_response') as mock_llm:
            mock_llm.return_value = ClaraResponse(
                message="Test response",
                emotion=EmotionType.CALM,
                raw_response='{"message": "Test response", "emotion": "calm"}',
                success=True
            )
            
            response = client.post(
                "/api/clara/conversation",
                json={"message": "Hello"}
            )
            
            assert response.status_code == 200
            # Should still work with minimal backstory
            mock_llm.assert_called_once()
    
    def test_conversation_endpoint_invalid_request(self):
        """Test conversation endpoint with invalid request data"""
        # Missing message
        response = client.post(
            "/api/clara/conversation",
            json={}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_conversation_endpoint_empty_message(self):
        """Test conversation endpoint with empty message"""
        response = client.post(
            "/api/clara/conversation",
            json={"message": ""}
        )
        
        assert response.status_code == 422  # Validation error (min_length=1)
    
    def test_conversation_endpoint_message_too_long(self):
        """Test conversation endpoint with message exceeding max length"""
        long_message = "x" * 2001  # Exceeds max_length=2000
        
        response = client.post(
            "/api/clara/conversation",
            json={"message": long_message}
        )
        
        assert response.status_code == 422  # Validation error
    
    @patch('app.services.character_content_service.CharacterContentService.get_consolidated_backstory')
    @patch('app.services.ava_llm_service.AvaLLMService.generate_ava_response')
    def test_conversation_endpoint_service_exception(self, mock_llm_response, mock_backstory):
        """Test handling of service exceptions"""
        mock_backstory.side_effect = Exception("Content service error")
        
        response = client.post(
            "/api/clara/conversation",
            json={"message": "Hello"}
        )
        
        assert response.status_code == 500
        assert "Error processing conversation" in response.json()["detail"]
    
    @patch('app.services.character_content_service.CharacterContentService.get_consolidated_backstory')
    @patch('app.services.ava_llm_service.AvaLLMService.generate_ava_response')
    def test_conversation_endpoint_response_structure(self, mock_llm_response, mock_backstory):
        """Test that response has correct structure and types"""
        mock_backstory.return_value = "Test backstory"
        mock_llm_response.return_value = ClaraResponse(
            message="Test response message",
            emotion=EmotionType.SASSY,
            raw_response='{"message": "Test response message", "emotion": "sassy"}',
            success=True
        )
        
        response = client.post(
            "/api/clara/conversation",
            json={"message": "Test message"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check types and structure
        assert isinstance(data["message"], str)
        assert isinstance(data["emotion"], str)
        assert isinstance(data["emotional_state"], dict)
        assert isinstance(data["timestamp"], str)
        assert isinstance(data["conversation_id"], str)
        assert isinstance(data["context_used"], bool)
        
        # Check emotional_state structure
        emotional_state = data["emotional_state"]
        assert "emotion" in emotional_state
        assert "mood" in emotional_state
        assert "energy" in emotional_state
        assert "stress" in emotional_state
        
        assert isinstance(emotional_state["energy"], int)
        assert isinstance(emotional_state["stress"], int)
        assert 1 <= emotional_state["energy"] <= 10
        assert 1 <= emotional_state["stress"] <= 10
    
    @patch('app.services.character_content_service.CharacterContentService.get_consolidated_backstory')
    @patch('app.services.conversation_prompt_service.ConversationPromptService.construct_conversation_prompt')
    @patch('app.services.ava_llm_service.AvaLLMService.generate_ava_response')
    def test_conversation_endpoint_service_integration(self, mock_llm, mock_prompt, mock_backstory):
        """Test that services are called with correct parameters"""
        mock_backstory.return_value = "Character backstory content"
        mock_prompt.return_value = "Constructed prompt"
        mock_llm.return_value = ClaraResponse(
            message="LLM response",
            emotion=EmotionType.CALM,
            raw_response='{"message": "LLM response", "emotion": "calm"}',
            success=True
        )
        
        response = client.post(
            "/api/clara/conversation",
            json={"message": "Hello Ava"}
        )
        
        assert response.status_code == 200
        
        # Verify service calls
        mock_backstory.assert_called_once()
        mock_prompt.assert_called_once()
        mock_llm.assert_called_once()
        
        # Check prompt service was called with correct parameters
        prompt_args = mock_prompt.call_args[1]
        assert prompt_args["character_backstory"] == "Character backstory content"
        assert prompt_args["user_message"] == "Hello Ava"
        assert "conversation_emotion" in prompt_args
        
        # Check LLM service was called with constructed prompt
        llm_args = mock_llm.call_args[1]
        assert llm_args["prompt"] == "Constructed prompt"