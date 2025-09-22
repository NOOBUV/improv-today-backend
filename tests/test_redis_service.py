import pytest
import uuid
from datetime import datetime, timezone
from unittest.mock import Mock, patch
from app.services.redis_service import RedisService
from app.models.conversation_v2 import ConversationMessage


class TestRedisService:
    """Test Redis service functionality including fallback behavior."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.redis_service = RedisService()
        self.conversation_id = str(uuid.uuid4())
        self.test_message = "Hello, this is a test message"
    
    @patch('app.services.redis_service.redis.from_url')
    def test_redis_connection_failure_fallback(self, mock_redis):
        """Test that Redis connection failure triggers database fallback (IV1)."""
        # Mock Redis connection failure
        mock_redis.side_effect = Exception("Connection failed")
        
        # Mock database session and query
        mock_db = Mock()
        mock_messages = [
            Mock(role="user", content="Hello", timestamp=datetime.now(timezone.utc)),
            Mock(role="assistant", content="Hi there!", timestamp=datetime.now(timezone.utc))
        ]
        mock_db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = mock_messages
        
        # Test conversation history retrieval with Redis unavailable
        history = self.redis_service.get_conversation_history(self.conversation_id, mock_db)
        
        # Verify fallback to database was used
        assert len(history) == 2
        # Messages are returned in chronological order after reversal
        assert history[0]["role"] == "assistant"  # First after reversal
        assert history[0]["content"] == "Hi there!"
        assert history[1]["role"] == "user"  # Second after reversal
        assert history[1]["content"] == "Hello"
        
        # Verify database was queried
        mock_db.query.assert_called_once()
    
    @patch('app.services.redis_service.redis.from_url')
    def test_cache_message_redis_unavailable(self, mock_redis):
        """Test message caching when Redis is unavailable."""
        # Mock Redis connection failure
        mock_redis.side_effect = Exception("Connection failed")
        
        # Test caching message when Redis unavailable
        result = self.redis_service.cache_message(
            self.conversation_id, 
            "user", 
            self.test_message
        )
        
        # Should return False when Redis unavailable
        assert result is False
    
    @patch('app.services.redis_service.redis.from_url')
    def test_successful_redis_operations(self, mock_redis):
        """Test successful Redis operations when connection is available."""
        # Mock successful Redis connection
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.lpush.return_value = 1
        mock_client.expire.return_value = True
        mock_client.ltrim.return_value = True
        mock_client.lrange.return_value = ['{"role": "user", "content": "Hello", "timestamp": "2023-01-01T00:00:00"}']
        mock_redis.return_value = mock_client
        
        # Test successful message caching
        result = self.redis_service.cache_message(
            self.conversation_id,
            "user", 
            self.test_message
        )
        
        assert result is True
        mock_client.lpush.assert_called_once()
        mock_client.expire.assert_called_once()
        mock_client.ltrim.assert_called_once()
    
    def test_build_conversation_context(self):
        """Test conversation context building from history data."""
        history_data = [
            {"role": "user", "content": "Hello there", "timestamp": "2023-01-01T00:00:00"},
            {"role": "assistant", "content": "Hi! How are you?", "timestamp": "2023-01-01T00:01:00"},
            {"role": "user", "content": "I'm doing well", "timestamp": "2023-01-01T00:02:00"}
        ]
        
        context = self.redis_service.build_conversation_context(history_data)
        
        expected_context = "User: Hello there\nAssistant: Hi! How are you?\nUser: I'm doing well"
        assert context == expected_context
    
    def test_build_conversation_context_empty(self):
        """Test conversation context building with empty history."""
        context = self.redis_service.build_conversation_context([])
        assert context == ""
    
    @patch('app.services.redis_service.redis.from_url')
    def test_health_check_success(self, mock_redis):
        """Test Redis health check when connection is healthy."""
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_redis.return_value = mock_client
        
        health = self.redis_service.health_check()
        
        assert health["connected"] is True
        assert health["ping_success"] is True
        assert "response_time_ms" in health
    
    @patch('app.services.redis_service.redis.from_url')
    def test_health_check_failure(self, mock_redis):
        """Test Redis health check when connection fails."""
        mock_redis.side_effect = Exception("Connection failed")
        
        health = self.redis_service.health_check()
        
        assert health["connected"] is False
        assert health["ping_success"] is False
    
    def test_conversation_history_message_limit(self):
        """Test that conversation history respects message limit (~15 messages)."""
        # Create mock database with more than 15 messages
        mock_db = Mock()
        mock_messages = []
        for i in range(20):
            mock_msg = Mock()
            mock_msg.role = "user" if i % 2 == 0 else "assistant"
            mock_msg.content = f"Message {i}"
            mock_msg.timestamp = datetime.now(timezone.utc)
            mock_messages.append(mock_msg)
        
        mock_db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = mock_messages[:15]
        
        # Mock Redis unavailable to force database fallback
        with patch('app.services.redis_service.redis.from_url', side_effect=Exception("Redis unavailable")):
            history = self.redis_service.get_conversation_history(self.conversation_id, mock_db, limit=15)
        
        # Verify limit was respected
        assert len(history) == 15
        
        # Verify limit parameter was passed to database query
        mock_db.query.return_value.filter.return_value.order_by.return_value.limit.assert_called_with(15)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])