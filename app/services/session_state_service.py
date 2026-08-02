"""
Session-specific state management service for per-user conversation context.
Manages session state storage with Redis and provides session lifecycle management.
"""

import logging
import json
import os
import re
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone, timedelta
from uuid import uuid4

from pydantic import BaseModel

from app.services.redis_service import RedisService
from app.services.simulation.state_manager import StateManagerService

logger = logging.getLogger(__name__)

# ponytail: hand-rolled stopword list beats pulling in nltk for one ILIKE query
_STOPWORDS = {
    "about", "actually", "after", "again", "already", "also", "always", "another",
    "anything", "around", "because", "been", "before", "being", "better", "could",
    "didn", "does", "doing", "done", "down", "even", "ever", "every", "from",
    "gonna", "good", "have", "havent", "hear", "here", "into", "just", "keep",
    "kind", "know", "like", "little", "look", "made", "make", "many", "maybe",
    "mean", "might", "more", "most", "much", "must", "need", "never", "next",
    "nice", "only", "other", "over", "people", "pretty", "really", "right",
    "said", "same", "should", "since", "some", "something", "still", "such",
    "sure", "take", "talk", "tell", "than", "that", "their", "them", "then",
    "there", "these", "they", "thing", "things", "think", "this", "those",
    "though", "time", "today", "told", "took", "very", "want", "well", "went",
    "were", "what", "when", "where", "which", "while", "with", "would", "your",
    "yeah", "your", "youre", "remember", "guess", "stuff", "back", "come",
    "came", "give", "getting", "goes", "going",
}

_SNIPPET_CHARS = 200

# Retrieval backend for get_related_past_snippets: "local" (keyword ILIKE over
# conversation_log) or "platform" (TencentDB-Agent-Memory L0 hybrid search).
# Platform errors/timeouts always fall back to local — conversation_log stays
# the source of truth either way.
CLARA_MEMORY_BACKEND = os.getenv("CLARA_MEMORY_BACKEND", "local")
CLARA_MEMORY_URL = os.getenv("CLARA_MEMORY_URL", "http://host.docker.internal:8420")
# Hot conversation path: budget is tight on purpose, fallback is cheap.
_PLATFORM_TIMEOUT_S = float(os.getenv("CLARA_MEMORY_TIMEOUT_S", "1.5"))

# "**[user]** Session: k [2026-08-02T15:53:26.913Z] (score: 0.033)\n\n<content>"
_PLATFORM_HIT = re.compile(
    r"\*\*\[(?P<role>\w+)\]\*\*[^\[]*\[(?P<ts>[^\]]+)\][^\n]*\n+(?P<content>.+?)(?=\n---|\Z)",
    re.DOTALL,
)


class PastSnippet(BaseModel):
    """One recalled line from an earlier conversation."""
    role: str  # "user" | "assistant"
    content: str
    age: str  # human phrasing, e.g. "yesterday", "3 days ago"


def _humanize_age(created_at: datetime) -> str:
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    days = (datetime.now(timezone.utc) - created_at).days
    if days <= 0:
        return "earlier today"
    if days == 1:
        return "yesterday"
    if days < 14:
        return f"{days} days ago"
    if days < 60:
        return f"{days // 7} weeks ago"
    return f"{days // 30} months ago"


def _keywords(message: str, limit: int = 6) -> List[str]:
    """Meaningful, de-duplicated words from the incoming message."""
    seen: List[str] = []
    for word in re.findall(r"[a-z']{4,}", message.lower()):
        word = word.strip("'")
        if len(word) >= 4 and word not in _STOPWORDS and word not in seen:
            seen.append(word)
    return seen[:limit]


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
                "conversation_messages": [],  # Store actual conversation messages
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

    async def add_conversation_message(
        self,
        user_id: str,
        conversation_id: str,
        message_type: str,  # "user" or "assistant"
        message_content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a conversation message to session state.

        Args:
            user_id: Unique identifier for the user
            conversation_id: Conversation session identifier
            message_type: Type of message ("user" or "assistant")
            message_content: Content of the message
            metadata: Additional metadata (sentiment, emotion, etc.)

        Returns:
            True if message added successfully
        """
        try:
            session_state = await self.get_session_state(user_id, conversation_id)

            if not session_state:
                # Create session if it doesn't exist
                result = await self.create_session_state(user_id, conversation_id)
                if not result.get("success"):
                    return False
                session_state = result.get("session_state")

            # Add message to conversation history
            message = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": message_type,
                "content": message_content,
                "metadata": metadata or {}
            }

            if "conversation_messages" not in session_state:
                session_state["conversation_messages"] = []

            session_state["conversation_messages"].append(message)

            # Update session metadata
            session_state["last_updated"] = datetime.now(timezone.utc).isoformat()
            session_state["session_metadata"]["total_interactions"] += 1
            session_state["session_metadata"]["last_activity"] = datetime.now(timezone.utc).isoformat()

            # Store updated state
            session_id = f"{user_id}:{conversation_id}"
            success = await self._store_session_state(session_id, session_state)

            await self._log_message_durably(
                user_id, conversation_id, message_type, message_content, metadata
            )

            logger.debug(f"Added {message_type} message to {session_id}")
            return success

        except Exception as e:
            logger.error(f"Error adding conversation message: {e}")
            return False

    async def _log_message_durably(
        self,
        user_id: str,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Append the turn to conversation_log. Redis session state expires after
        session_ttl; this row does not.

        Best-effort: logging must never fail a conversation turn.
        """
        try:
            from app.core.database import AsyncSessionLocal
            from app.models.conversation_log import ConversationLog

            async with AsyncSessionLocal() as session:
                session.add(ConversationLog(
                    user_id=str(user_id),
                    conversation_id=str(conversation_id),
                    role=role,
                    content=content,
                    meta=metadata or None,
                ))
                await session.commit()
        except Exception as e:
            logger.warning(f"conversation_log write failed (turn unaffected): {e}")

    async def get_related_past_snippets(
        self,
        user_id: str,
        user_message: str,
        exclude_conversation_id: str,
        limit: int = 3
    ) -> List[PastSnippet]:
        """
        Associative recall, dispatched on CLARA_MEMORY_BACKEND.

        Default "local" is the keyword query below. "platform" asks
        TencentDB-Agent-Memory for hybrid (vector + FTS) hits and falls back to
        local on any error or timeout.
        """
        if CLARA_MEMORY_BACKEND == "platform":
            snippets = await self._platform_past_snippets(user_message, limit)
            if snippets is not None:
                return snippets
        return await self._local_past_snippets(
            user_id, user_message, exclude_conversation_id, limit
        )

    async def _platform_past_snippets(
        self, user_message: str, limit: int
    ) -> Optional[List[PastSnippet]]:
        """
        L0 hybrid search against the memory platform.

        Returns None (not []) on failure so the caller can tell "platform is
        down, use local" apart from "platform ran and found nothing".
        """
        try:
            import httpx

            async with httpx.AsyncClient(timeout=_PLATFORM_TIMEOUT_S) as client:
                resp = await client.post(
                    f"{CLARA_MEMORY_URL}/search/conversations",
                    json={"query": user_message, "limit": limit},
                    headers={"x-tdai-service-id": "default"},
                )
                resp.raise_for_status()
                results = resp.json().get("results", "")

            snippets: List[PastSnippet] = []
            for m in _PLATFORM_HIT.finditer(results):
                content = m.group("content").strip()
                if not content:
                    continue
                try:
                    ts = datetime.fromisoformat(m.group("ts").replace("Z", "+00:00"))
                    age = _humanize_age(ts)
                except ValueError:
                    age = "earlier"
                snippets.append(PastSnippet(
                    role=m.group("role"),
                    content=(
                        content[:_SNIPPET_CHARS].rstrip() + "..."
                        if len(content) > _SNIPPET_CHARS else content
                    ),
                    age=age,
                ))
                if len(snippets) >= limit:
                    break
            return snippets

        except Exception as e:
            logger.warning(f"Platform recall failed, falling back to local: {e}")
            return None

    async def _local_past_snippets(
        self,
        user_id: str,
        user_message: str,
        exclude_conversation_id: str,
        limit: int = 3
    ) -> List[PastSnippet]:
        """
        Lines from OTHER conversations that share keywords with the incoming
        message, most-matching first, then most recent.

        ponytail: ILIKE keyword overlap, no embeddings. Swap in pg_trgm/tsvector or
        a vector index if recall quality becomes the bottleneck.

        Best-effort: any failure returns [] so the turn proceeds without memories.
        """
        words = _keywords(user_message)
        if not words:
            return []

        try:
            from sqlalchemy import select, or_, case, desc
            from app.core.database import AsyncSessionLocal
            from app.models.conversation_log import ConversationLog

            matches = [ConversationLog.content.ilike(f"%{w}%") for w in words]
            score = sum(case((m, 1), else_=0) for m in matches)

            stmt = (
                select(ConversationLog, score.label("score"))
                .where(
                    ConversationLog.user_id == str(user_id),
                    ConversationLog.conversation_id != str(exclude_conversation_id),
                    or_(*matches),
                )
                .order_by(desc("score"), desc(ConversationLog.created_at))
                .limit(limit)
            )

            async with AsyncSessionLocal() as session:
                rows = (await session.execute(stmt)).all()

            return [
                PastSnippet(
                    role=row[0].role,
                    content=(
                        row[0].content[:_SNIPPET_CHARS].rstrip() + "..."
                        if len(row[0].content) > _SNIPPET_CHARS
                        else row[0].content
                    ),
                    age=_humanize_age(row[0].created_at),
                )
                for row in rows
            ]

        except Exception as e:
            logger.warning(f"Past-conversation recall failed (turn unaffected): {e}")
            return []

    async def get_conversation_history(
        self,
        user_id: str,
        conversation_id: str,
        max_messages: int = 12
    ) -> str:
        """
        Get formatted conversation history for prompt context.

        Args:
            user_id: Unique identifier for the user
            conversation_id: Conversation session identifier
            max_messages: Maximum number of recent messages to include

        Returns:
            Formatted conversation history string
        """
        try:
            session_state = await self.get_session_state(user_id, conversation_id)

            if not session_state or "conversation_messages" not in session_state:
                return ""

            messages = session_state["conversation_messages"]

            # Get recent messages (up to max_messages)
            recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages

            # Format for conversation context
            history_parts = []
            for msg in recent_messages:
                role = "Human" if msg["type"] == "user" else "Clara"
                content = msg["content"]
                history_parts.append(f"{role}: {content}")

            return "\n".join(history_parts)

        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return ""

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