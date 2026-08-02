import pytest
import uuid
from unittest.mock import Mock, patch
from app.services.redis_service import RedisService


class TestRedisService:
    """Test Redis service functionality including fallback behavior."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.redis_service = RedisService()
        self.conversation_id = str(uuid.uuid4())
        self.test_message = "Hello, this is a test message"
    
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])