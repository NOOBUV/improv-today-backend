import json
import redis
from typing import List, Dict, Optional
from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session
from app.core.config import settings
from app.models.conversation_v2 import ConversationMessage
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
    
    def get_conversation_history(self, conversation_id: str, db: Session, limit: int = 15) -> List[Dict]:
        """
        Retrieve conversation history from Redis with database fallback.
        
        Args:
            conversation_id: UUID of the conversation
            db: Database session for fallback queries
            limit: Maximum number of messages to retrieve (~15 as per AC: 1)
            
        Returns:
            List of message dictionaries with role, content, timestamp
        """
        # Try Redis first
        client = self._get_client()
        if client:
            try:
                cache_key = f"conversation_history:{conversation_id}"
                cached_messages = client.lrange(cache_key, 0, limit - 1)
                
                if cached_messages:
                    messages = []
                    for msg_json in reversed(cached_messages):  # Reverse for chronological order
                        try:
                            msg_data = json.loads(msg_json)
                            messages.append({
                                "role": msg_data["role"],
                                "content": msg_data["content"],
                                "timestamp": msg_data["timestamp"]
                            })
                        except json.JSONDecodeError:
                            logger.warning(f"⚠️ Failed to parse cached message: {msg_json}")
                            continue
                    
                    logger.debug(f"✅ Retrieved {len(messages)} messages from Redis cache")
                    return messages
                    
            except Exception as e:
                logger.error(f"❌ Redis retrieval failed: {str(e)}. Falling back to database.")
        
        # Fallback to database (AC: IV1)
        return self._get_history_from_database(conversation_id, db, limit)
    
    def _get_history_from_database(self, conversation_id: str, db: Session, limit: int) -> List[Dict]:
        """
        Fallback method to retrieve conversation history from database.
        
        Args:
            conversation_id: UUID of the conversation
            db: Database session
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of message dictionaries from database
        """
        try:
            if db is None:
                logger.warning("⚠️ Database session is None, cannot retrieve conversation history")
                return []
                
            messages = db.query(ConversationMessage).filter(
                ConversationMessage.conversation_id == conversation_id
            ).order_by(
                ConversationMessage.timestamp.desc()
            ).limit(limit).all()
            
            # Convert to list and reverse for chronological order
            history = []
            for msg in reversed(messages):
                history.append({
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat()
                })
            
            logger.info(f"📁 Retrieved {len(history)} messages from database fallback")
            return history
            
        except Exception as e:
            logger.error(f"❌ Database fallback failed: {str(e)}")
            return []
    
    def build_conversation_context(self, conversation_history: List[Dict]) -> str:
        """
        Build conversation context string from message history for OpenAI prompts.
        
        Args:
            conversation_history: List of message dictionaries
            
        Returns:
            Formatted conversation context string for prompt injection
        """
        if not conversation_history:
            return ""
        
        context_lines = []
        for msg in conversation_history:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            context_lines.append(f"{role_label}: {msg['content']}")
        
        return "\n".join(context_lines)
    
    def clear_conversation_cache(self, conversation_id: str) -> bool:
        """
        Clear cached conversation history for a specific conversation.
        
        Args:
            conversation_id: UUID of the conversation to clear
            
        Returns:
            bool: True if cleared successfully or Redis unavailable
        """
        client = self._get_client()
        if not client:
            return True  # No cache to clear
            
        try:
            cache_key = f"conversation_history:{conversation_id}"
            client.delete(cache_key)
            logger.debug(f"🗑️ Cleared cache for conversation {conversation_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to clear cache: {str(e)}")
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