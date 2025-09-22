"""
Session-specific state management service for per-user conversation context.
Manages session state storage with Redis and provides session lifecycle management.
"""

import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone, timedelta
from uuid import uuid4

from app.services.redis_service import RedisService
from app.services.simulation.state_manager import StateManagerService

logger = logging.getLogger(__name__)


class SessionStateService:
    """
    Service for managing per-user conversation context and session-specific state.

    This service handles:
    - Session state creation and initialization from global state
    - Per-user personalization data storage
    - Session lifecycle management (creation, updates, expiration, cleanup)
    - Session state isolation between users
    """

    def __init__(self):
        self.redis_service = RedisService()
        self.state_manager = StateManagerService()
        self.session_ttl = 86400  # 24 hours in seconds
        self.session_key_prefix = "session_state"

    async def create_session_state(
        self,
        user_id: str,
        conversation_id: str,
        personalization_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create new session state initialized from global state.

        Args:
            user_id: Unique identifier for the user
            conversation_id: Conversation session identifier
            personalization_data: User-specific preferences and context

        Returns:
            Created session state data
        """
        try:
            # Get current global state as baseline
            global_state = await self.state_manager.get_current_global_state()

            # Create session ID
            session_id = f"{user_id}:{conversation_id}"

            # Initialize session state from global state
            session_state = {
                "session_id": session_id,
                "user_id": user_id,
                "conversation_id": conversation_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "global_state_baseline": global_state,
                "session_adjustments": {},  # User-specific state modifications
                "personalization": personalization_data or {},
                "conversation_context": {
                    "relationship_level": "new",
                    "preferred_communication_style": "balanced",
                    "user_mood_indicators": [],
                    "conversation_tone": "neutral"
                },
                "session_metadata": {
                    "total_interactions": 0,
                    "session_duration_minutes": 0,
                    "last_activity": datetime.now(timezone.utc).isoformat()
                }
            }

            # Store in Redis
            success = await self._store_session_state(session_id, session_state)

            if success:
                logger.info(f"Created session state for user {user_id}, conversation {conversation_id}")
                return {
                    "success": True,
                    "session_id": session_id,
                    "session_state": session_state
                }
            else:
                logger.error(f"Failed to store session state for {session_id}")
                return {
                    "success": False,
                    "error": "Failed to store session state"
                }

        except Exception as e:
            logger.error(f"Error creating session state: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def get_session_state(
        self,
        user_id: str,
        conversation_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve session state for a specific user conversation.

        Args:
            user_id: Unique identifier for the user
            conversation_id: Conversation session identifier

        Returns:
            Session state data or None if not found
        """
        try:
            session_id = f"{user_id}:{conversation_id}"

            # Try to get from Redis
            session_state = await self._get_session_state(session_id)

            if session_state:
                # Update last activity
                session_state["session_metadata"]["last_activity"] = datetime.now(timezone.utc).isoformat()
                await self._store_session_state(session_id, session_state)

                logger.debug(f"Retrieved session state for {session_id}")
                return session_state
            else:
                logger.debug(f"No session state found for {session_id}")
                return None

        except Exception as e:
            logger.error(f"Error getting session state: {e}")
            return None

    async def update_session_adjustments(
        self,
        user_id: str,
        conversation_id: str,
        trait_adjustments: Dict[str, Any]
    ) -> bool:
        """
        Update session-specific state adjustments based on conversation.

        Args:
            user_id: Unique identifier for the user
            conversation_id: Conversation session identifier
            trait_adjustments: Dictionary of trait names and adjustment values

        Returns:
            True if updated successfully
        """
        try:
            session_state = await self.get_session_state(user_id, conversation_id)

            if not session_state:
                logger.warning(f"No session state found for user {user_id}, conversation {conversation_id}")
                return False

            # Update session adjustments
            if "session_adjustments" not in session_state:
                session_state["session_adjustments"] = {}

            for trait_name, adjustment in trait_adjustments.items():
                session_state["session_adjustments"][trait_name] = {
                    "value": adjustment.get("value", 0),
                    "reason": adjustment.get("reason", "Session-specific adjustment"),
                    "updated_at": datetime.now(timezone.utc).isoformat()
                }

            # Update metadata
            session_state["last_updated"] = datetime.now(timezone.utc).isoformat()
            session_state["session_metadata"]["total_interactions"] += 1

            # Store updated state
            session_id = f"{user_id}:{conversation_id}"
            success = await self._store_session_state(session_id, session_state)

            if success:
                logger.info(f"Updated session adjustments for {session_id}: {len(trait_adjustments)} traits")
                return True
            else:
                logger.error(f"Failed to store updated session state for {session_id}")
                return False

        except Exception as e:
            logger.error(f"Error updating session adjustments: {e}")
            return False

    async def update_conversation_context(
        self,
        user_id: str,
        conversation_id: str,
        context_updates: Dict[str, Any]
    ) -> bool:
        """
        Update conversation context within session state.

        Args:
            user_id: Unique identifier for the user
            conversation_id: Conversation session identifier
            context_updates: Dictionary of context updates

        Returns:
            True if updated successfully
        """
        try:
            session_state = await self.get_session_state(user_id, conversation_id)

            if not session_state:
                return False

            # Update conversation context
            if "conversation_context" not in session_state:
                session_state["conversation_context"] = {}

            session_state["conversation_context"].update(context_updates)
            session_state["last_updated"] = datetime.now(timezone.utc).isoformat()

            # Store updated state
            session_id = f"{user_id}:{conversation_id}"
            success = await self._store_session_state(session_id, session_state)

            logger.debug(f"Updated conversation context for {session_id}")
            return success

        except Exception as e:
            logger.error(f"Error updating conversation context: {e}")
            return False

    async def get_effective_state(
        self,
        user_id: str,
        conversation_id: str
    ) -> Dict[str, Any]:
        """
        Get effective state by merging global state with session adjustments.

        Args:
            user_id: Unique identifier for the user
            conversation_id: Conversation session identifier

        Returns:
            Effective state combining global baseline and session adjustments
        """
        try:
            session_state = await self.get_session_state(user_id, conversation_id)

            if not session_state:
                # Fallback to current global state
                logger.info(f"No session state, using global state for user {user_id}")
                return await self.state_manager.get_current_global_state()

            # Start with global state baseline
            effective_state = session_state.get("global_state_baseline", {}).copy()

            # Apply session adjustments
            session_adjustments = session_state.get("session_adjustments", {})

            for trait_name, adjustment in session_adjustments.items():
                if trait_name in effective_state:
                    # Apply adjustment to baseline value
                    baseline_value = effective_state[trait_name].get("numeric_value", 50)
                    adjustment_value = adjustment.get("value", 0)

                    # Calculate new effective value (keep within bounds)
                    new_value = max(0, min(100, baseline_value + adjustment_value))

                    # Update effective state
                    effective_state[trait_name]["numeric_value"] = new_value
                    effective_state[trait_name]["session_adjusted"] = True
                    effective_state[trait_name]["adjustment_reason"] = adjustment.get("reason", "")

            logger.debug(f"Calculated effective state for {user_id} with {len(session_adjustments)} adjustments")
            return effective_state

        except Exception as e:
            logger.error(f"Error getting effective state: {e}")
            # Fallback to global state
            return await self.state_manager.get_current_global_state()

    async def expire_session(
        self,
        user_id: str,
        conversation_id: str
    ) -> bool:
        """
        Manually expire a session and clean up its state.

        Args:
            user_id: Unique identifier for the user
            conversation_id: Conversation session identifier

        Returns:
            True if expired successfully
        """
        try:
            session_id = f"{user_id}:{conversation_id}"

            # Delete from Redis
            client = self.redis_service._get_client()
            if client:
                cache_key = f"{self.session_key_prefix}:{session_id}"
                client.delete(cache_key)
                logger.info(f"Expired session state for {session_id}")
                return True
            else:
                logger.warning("Redis unavailable, cannot expire session")
                return False

        except Exception as e:
            logger.error(f"Error expiring session: {e}")
            return False

    async def list_active_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """
        List all active sessions for a user.

        Args:
            user_id: Unique identifier for the user

        Returns:
            List of active session summaries
        """
        try:
            client = self.redis_service._get_client()
            if not client:
                logger.warning("Redis unavailable, cannot list sessions")
                return []

            # Find all session keys for this user
            pattern = f"{self.session_key_prefix}:{user_id}:*"
            session_keys = client.keys(pattern)

            active_sessions = []
            for key in session_keys:
                try:
                    session_data = client.get(key)
                    if session_data:
                        state = json.loads(session_data)
                        active_sessions.append({
                            "session_id": state.get("session_id"),
                            "conversation_id": state.get("conversation_id"),
                            "created_at": state.get("created_at"),
                            "last_updated": state.get("last_updated"),
                            "total_interactions": state.get("session_metadata", {}).get("total_interactions", 0)
                        })
                except Exception as e:
                    logger.warning(f"Error parsing session data for key {key}: {e}")
                    continue

            logger.debug(f"Found {len(active_sessions)} active sessions for user {user_id}")
            return active_sessions

        except Exception as e:
            logger.error(f"Error listing active sessions: {e}")
            return []

    async def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions based on TTL.

        Returns:
            Number of sessions cleaned up
        """
        try:
            client = self.redis_service._get_client()
            if not client:
                return 0

            # Find all session keys
            pattern = f"{self.session_key_prefix}:*"
            session_keys = client.keys(pattern)

            cleaned_count = 0
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)

            for key in session_keys:
                try:
                    session_data = client.get(key)
                    if session_data:
                        state = json.loads(session_data)
                        last_activity = datetime.fromisoformat(
                            state.get("session_metadata", {}).get("last_activity", "")
                        )

                        if last_activity < cutoff_time:
                            client.delete(key)
                            cleaned_count += 1

                except Exception as e:
                    logger.warning(f"Error processing session key {key} during cleanup: {e}")
                    continue

            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} expired sessions")

            return cleaned_count

        except Exception as e:
            logger.error(f"Error during session cleanup: {e}")
            return 0

    async def _store_session_state(
        self,
        session_id: str,
        session_state: Dict[str, Any]
    ) -> bool:
        """Store session state in Redis with TTL."""
        try:
            client = self.redis_service._get_client()
            if not client:
                logger.warning("Redis unavailable, cannot store session state")
                return False

            cache_key = f"{self.session_key_prefix}:{session_id}"
            serialized_state = json.dumps(session_state, default=str)

            # Store with TTL
            client.setex(cache_key, self.session_ttl, serialized_state)
            return True

        except Exception as e:
            logger.error(f"Error storing session state: {e}")
            return False

    async def _get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve session state from Redis."""
        try:
            client = self.redis_service._get_client()
            if not client:
                return None

            cache_key = f"{self.session_key_prefix}:{session_id}"
            session_data = client.get(cache_key)

            if session_data:
                return json.loads(session_data)
            return None

        except Exception as e:
            logger.error(f"Error retrieving session state: {e}")
            return None