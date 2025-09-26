"""
Event Usage Tracking Service using Redis for conversation variety.
Prevents Clara from repeating the same life events to the same user.
"""

import logging
import json
import asyncio
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
import redis.asyncio as redis
from app.core.config import settings

logger = logging.getLogger(__name__)


class EventUsageTracker:
    """
    Redis-based service to track which life events Clara has mentioned to each user.
    Ensures conversation variety by rotating through different events.
    """

    def __init__(self):
        self.redis_client = None
        self._connection_initialized = False

        # Redis key patterns
        self.USER_EVENTS_KEY = "clara:user_events:{user_id}"
        self.CONVERSATION_EVENTS_KEY = "clara:conv_events:{conversation_id}"
        self.EVENT_USAGE_STATS_KEY = "clara:event_stats"

        # Cache TTL settings (in seconds)
        self.USER_EVENTS_TTL = 30 * 24 * 3600  # 30 days
        self.CONVERSATION_EVENTS_TTL = 7 * 24 * 3600  # 7 days

    async def _ensure_connection(self):
        """Ensure Redis connection is established"""
        if not self._connection_initialized:
            try:
                redis_url = getattr(settings, 'redis_url', 'redis://localhost:6379')
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                # Test connection
                await self.redis_client.ping()
                self._connection_initialized = True
                logger.info("Redis connection established for EventUsageTracker")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                # Fallback to in-memory tracking
                self.redis_client = None
                self._in_memory_fallback = {}
                logger.warning("Falling back to in-memory event tracking")

    async def track_events_used(
        self,
        user_id: str,
        conversation_id: str,
        events_used: List[str]
    ) -> bool:
        """
        Track which events Clara mentioned in this response.

        Args:
            user_id: User identifier
            conversation_id: Conversation session ID
            events_used: List of event IDs that were mentioned

        Returns:
            bool: Success status
        """
        try:
            await self._ensure_connection()

            if not events_used:
                return True

            if self.redis_client is None:
                # Fallback to in-memory
                return self._track_events_fallback(user_id, conversation_id, events_used)

            timestamp = datetime.now().isoformat()

            # Update user-level event usage (cross-conversation tracking)
            user_key = self.USER_EVENTS_KEY.format(user_id=user_id)
            for event_id in events_used:
                event_data = {
                    "event_id": event_id,
                    "last_used": timestamp,
                    "conversation_id": conversation_id,
                    "usage_count": 1
                }

                # Check if event was used before by this user
                existing_data = await self.redis_client.hget(user_key, event_id)
                if existing_data:
                    existing = json.loads(existing_data)
                    event_data["usage_count"] = existing.get("usage_count", 0) + 1

                await self.redis_client.hset(
                    user_key,
                    event_id,
                    json.dumps(event_data)
                )

            # Set TTL for user events
            await self.redis_client.expire(user_key, self.USER_EVENTS_TTL)

            # Update conversation-level event usage
            conv_key = self.CONVERSATION_EVENTS_KEY.format(conversation_id=conversation_id)
            conv_data = {
                "timestamp": timestamp,
                "events": events_used,
                "user_id": user_id
            }
            await self.redis_client.lpush(conv_key, json.dumps(conv_data))
            await self.redis_client.expire(conv_key, self.CONVERSATION_EVENTS_TTL)

            # Update global event usage statistics
            for event_id in events_used:
                await self.redis_client.hincrby(self.EVENT_USAGE_STATS_KEY, event_id, 1)

            logger.debug(f"Tracked {len(events_used)} events for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Error tracking events: {e}")
            return False

    async def get_fresh_events(
        self,
        user_id: str,
        event_pool: List[Dict[str, Any]],
        max_events: int = 3,
        avoid_recent_days: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Get events that haven't been used recently with this user.

        Args:
            user_id: User identifier
            event_pool: Available events to choose from
            max_events: Maximum number of events to return
            avoid_recent_days: Don't reuse events used in last N days

        Returns:
            List of fresh events
        """
        try:
            await self._ensure_connection()

            if self.redis_client is None:
                return self._get_fresh_events_fallback(user_id, event_pool, max_events)

            # Get user's recent event usage
            user_key = self.USER_EVENTS_KEY.format(user_id=user_id)
            user_events = await self.redis_client.hgetall(user_key)

            # Filter out recently used events
            cutoff_date = datetime.now() - timedelta(days=avoid_recent_days)
            recently_used = set()

            for event_id, event_data_str in user_events.items():
                try:
                    event_data = json.loads(event_data_str)
                    last_used = datetime.fromisoformat(event_data["last_used"])
                    if last_used > cutoff_date:
                        recently_used.add(event_id)
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue

            # Select fresh events
            fresh_events = [
                event for event in event_pool
                if event.get("id", "") not in recently_used
            ]

            # If we don't have enough fresh events, include some older ones
            if len(fresh_events) < max_events:
                older_events = [
                    event for event in event_pool
                    if event.get("id", "") in recently_used
                ]
                # Sort by last usage (oldest first)
                older_events.sort(key=lambda e: self._get_last_usage_time(user_events, e.get("id", "")))
                fresh_events.extend(older_events[:max_events - len(fresh_events)])

            result = fresh_events[:max_events]
            logger.debug(f"Selected {len(result)} fresh events for user {user_id}")
            return result

        except Exception as e:
            logger.error(f"Error getting fresh events: {e}")
            # Return random events as fallback
            return event_pool[:max_events]

    async def get_user_event_history(self, user_id: str) -> Dict[str, Any]:
        """
        Get user's event usage history for debugging/analysis.

        Args:
            user_id: User identifier

        Returns:
            Dictionary with user's event usage statistics
        """
        try:
            await self._ensure_connection()

            if self.redis_client is None:
                return {"error": "Redis not available"}

            user_key = self.USER_EVENTS_KEY.format(user_id=user_id)
            user_events = await self.redis_client.hgetall(user_key)

            history = {}
            for event_id, event_data_str in user_events.items():
                try:
                    event_data = json.loads(event_data_str)
                    history[event_id] = event_data
                except json.JSONDecodeError:
                    continue

            return {
                "user_id": user_id,
                "total_events_used": len(history),
                "events": history
            }

        except Exception as e:
            logger.error(f"Error getting user event history: {e}")
            return {"error": str(e)}

    async def get_event_usage_stats(self) -> Dict[str, int]:
        """Get global event usage statistics"""
        try:
            await self._ensure_connection()

            if self.redis_client is None:
                return {}

            stats = await self.redis_client.hgetall(self.EVENT_USAGE_STATS_KEY)
            return {event_id: int(count) for event_id, count in stats.items()}

        except Exception as e:
            logger.error(f"Error getting event usage stats: {e}")
            return {}

    def _get_last_usage_time(self, user_events: Dict[str, str], event_id: str) -> datetime:
        """Get last usage time for an event, return very old date if not found"""
        try:
            if event_id in user_events:
                event_data = json.loads(user_events[event_id])
                return datetime.fromisoformat(event_data["last_used"])
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

        return datetime.min  # Very old date

    # Fallback methods for when Redis is not available
    def _track_events_fallback(self, user_id: str, conversation_id: str, events_used: List[str]) -> bool:
        """In-memory fallback for event tracking"""
        if not hasattr(self, '_in_memory_fallback'):
            self._in_memory_fallback = {}

        if user_id not in self._in_memory_fallback:
            self._in_memory_fallback[user_id] = set()

        self._in_memory_fallback[user_id].update(events_used)
        return True

    def _get_fresh_events_fallback(self, user_id: str, event_pool: List[Dict[str, Any]], max_events: int) -> List[Dict[str, Any]]:
        """In-memory fallback for getting fresh events"""
        if not hasattr(self, '_in_memory_fallback'):
            self._in_memory_fallback = {}

        used_events = self._in_memory_fallback.get(user_id, set())
        fresh_events = [
            event for event in event_pool
            if event.get("id", "") not in used_events
        ]

        if len(fresh_events) < max_events:
            fresh_events.extend(event_pool[:max_events - len(fresh_events)])

        return fresh_events[:max_events]

    async def cleanup_old_data(self, days_old: int = 30) -> Dict[str, int]:
        """
        Cleanup old event usage data.

        Args:
            days_old: Remove data older than this many days

        Returns:
            Statistics about cleaned data
        """
        try:
            await self._ensure_connection()

            if self.redis_client is None:
                return {"error": "Redis not available"}

            cutoff_date = datetime.now() - timedelta(days=days_old)
            cleaned_users = 0
            cleaned_conversations = 0

            # Clean user event data
            async for key in self.redis_client.scan_iter(match=self.USER_EVENTS_KEY.format(user_id="*")):
                user_events = await self.redis_client.hgetall(key)
                events_to_remove = []

                for event_id, event_data_str in user_events.items():
                    try:
                        event_data = json.loads(event_data_str)
                        last_used = datetime.fromisoformat(event_data["last_used"])
                        if last_used < cutoff_date:
                            events_to_remove.append(event_id)
                    except (json.JSONDecodeError, KeyError, ValueError):
                        events_to_remove.append(event_id)

                if events_to_remove:
                    await self.redis_client.hdel(key, *events_to_remove)
                    cleaned_users += 1

            # Clean conversation data (automatically expires, but manual cleanup for immediate effect)
            async for key in self.redis_client.scan_iter(match=self.CONVERSATION_EVENTS_KEY.format(conversation_id="*")):
                await self.redis_client.delete(key)
                cleaned_conversations += 1

            return {
                "cleaned_users": cleaned_users,
                "cleaned_conversations": cleaned_conversations,
                "cutoff_date": cutoff_date.isoformat()
            }

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return {"error": str(e)}