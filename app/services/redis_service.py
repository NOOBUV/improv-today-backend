import json
import redis
from typing import Dict, Optional
from datetime import datetime, timezone
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


class RedisService:
    """
    Redis service for caching conversation history with database fallback.
    
    Implements the cache key pattern: conversation_history:{conversation_id}
    Provides fallback to database when Redis is unavailable (AC: IV1).
    """
    
    def __init__(self):
        """Initialize Redis client with connection handling."""
        self._client: Optional[redis.Redis] = None
        self._connection_tested = False
        
    def _get_client(self) -> Optional[redis.Redis]:
        """
        Get Redis client with lazy initialization and connection testing.
        
        Returns:
            Redis client if available, None if connection fails
        """
        if self._client is None or not self._connection_tested:
            try:
                self._client = redis.from_url(
                    settings.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                    health_check_interval=30
                )
                # Test connection
                self._client.ping()
                self._connection_tested = True
                logger.info("✅ Redis connection established successfully")
                return self._client
            except Exception as e:
                logger.warning(f"⚠️ Redis connection failed: {str(e)}. Falling back to database.")
                self._client = None
                self._connection_tested = False
                return None
        return self._client
    
    def cache_message(self, conversation_id: str, role: str, content: str, timestamp: Optional[datetime] = None) -> bool:
        """
        Cache a conversation message in Redis.
        
        Args:
            conversation_id: UUID of the conversation
            role: Message role ('user' or 'assistant') 
            content: Message content
            timestamp: Message timestamp (defaults to now)
            
        Returns:
            bool: True if cached successfully, False if Redis unavailable
        """
        client = self._get_client()
        if not client:
            return False
            
        try:
            if timestamp is None:
                timestamp = datetime.now(timezone.utc)
                
            message_data = {
                "role": role,
                "content": content,
                "timestamp": timestamp.isoformat()
            }
            
            cache_key = f"conversation_history:{conversation_id}"
            
            # Add message to list (LPUSH for chronological order)
            client.lpush(cache_key, json.dumps(message_data))
            
            # Set TTL for automatic cleanup (24 hours)
            client.expire(cache_key, 86400)
            
            # Keep only last 30 messages (double the ~15 limit for safety)
            client.ltrim(cache_key, 0, 29)
            
            logger.debug(f"✅ Cached message for conversation {conversation_id}: {role}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to cache message: {str(e)}")
            return False
    
    def health_check(self) -> Dict[str, bool]:
        """
        Check Redis service health.
        
        Returns:
            Dictionary with connection status and response time
        """
        client = self._get_client()
        if not client:
            return {"connected": False, "ping_success": False}
            
        try:
            start_time = datetime.now(timezone.utc)
            client.ping()
            response_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            return {
                "connected": True,
                "ping_success": True,
                "response_time_ms": response_time
            }
        except Exception as e:
            logger.error(f"❌ Redis health check failed: {str(e)}")
            return {"connected": False, "ping_success": False, "error": str(e)}