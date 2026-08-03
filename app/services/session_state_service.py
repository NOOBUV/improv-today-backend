"""
Session-specific state management service for per-user conversation context.
Manages session state storage with Redis and provides session lifecycle management.
"""

import asyncio
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

# Cosine-similarity floor for vector recall. Returning fewer than `limit`
# snippets is the correct outcome, not a bug — a vague message should recall
# nothing rather than pad the prompt with the nearest line.
#
# Tuned on user 33's history: gemini-embedding-001 compresses this corpus into
# roughly 0.48-0.72, so the floor lives in a narrow band. Off-topic queries
# ("what's the weather in tokyo") top out at 0.49 and are cut cleanly; the
# weakest hit that MUST survive is the disaster message at 0.62 against "the day
# everything fell apart in the morning". Hence 0.60 — raising it to 0.62 loses
# that recall, which is why the length filter below, not this number, does the
# heavy lifting against padding.
RECALL_SIMILARITY_FLOOR = float(os.getenv("CLARA_RECALL_FLOOR", "0.60"))

# Minimum row length to be recallable, and the more important of the two knobs.
# Short generic lines sit near the centre of the embedding space and score high
# against everything: on user 33's history "morning" scored 0.70 and "what
# happened" 0.64 against "the day everything fell apart in the morning", while
# the actual 341-char disaster message scored 0.62. No similarity floor can
# separate those — length can, because a line worth recalling months later is
# never seven characters long.
# ponytail: length as a proxy for "has content". If a short-but-meaningful line
# ("I got the job") ever needs recalling, filter on token variety instead.
MIN_RECALL_CHARS = 40

# Similarity bonus for consolidated role='memory' rows.
# Not a thumb on the scale: a nightly memory is deliberately generic third-person
# prose ("The user shared that his project situation improved"), so it loses
# cosine to any verbatim line that happens to echo the user's own words — on the
# demo query it scored 0.659 against 0.730 for the raw quote it was distilled
# from. Without this, the rows we spend an LLM call producing are structurally
# outranked by the rows they summarise.
MEMORY_SIMILARITY_BOOST = 0.05


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


async def embed_logged_row(row_id: int, content: str) -> None:
    """Attach an embedding to a freshly written conversation_log row. Best-effort."""
    try:
        from sqlalchemy import text as sql_text
        from app.core.database import AsyncSessionLocal
        from app.services.embeddings import embed_one, to_pgvector

        # Don't spend an embedding call on a database that has nowhere to put the
        # result (sqlite in tests, or any non-postgres deployment).
        bind = getattr(AsyncSessionLocal, "kw", {}).get("bind")
        if bind is not None and bind.dialect.name != "postgresql":
            return

        vector = await embed_one(content)
        if vector is None:
            return
        async with AsyncSessionLocal() as session:
            await session.execute(
                sql_text(
                    "UPDATE conversation_log SET embedding = CAST(:v AS vector) WHERE id = :id"
                ),
                {"v": to_pgvector(vector), "id": row_id},
            )
            await session.commit()
    except Exception as e:
        logger.warning(f"Embedding write failed for conversation_log {row_id}: {e}")


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
                row = ConversationLog(
                    user_id=str(user_id),
                    conversation_id=str(conversation_id),
                    role=role,
                    content=content,
                    meta=metadata or None,
                )
                session.add(row)
                await session.commit()
                row_id = row.id
        except Exception as e:
            logger.warning(f"conversation_log write failed (turn unaffected): {e}")
            return

        # ponytail: fire-and-forget. The embedding costs ~450ms and this is the hot
        # turn path; a dropped task just leaves the row NULL for the backfill script.
        try:
            asyncio.get_running_loop().create_task(embed_logged_row(row_id, content))
        except RuntimeError:
            pass

    async def get_related_past_snippets(
        self,
        user_id: str,
        user_message: str,
        exclude_conversation_id: str,
        limit: int = 3
    ) -> List[PastSnippet]:
        """
        Associative recall from earlier conversations.

        Semantic first: embed the incoming message and cosine-search the rows
        that carry an embedding. Anything under RECALL_SIMILARITY_FLOOR is
        dropped, so a vague message legitimately recalls nothing rather than
        padding the prompt with the nearest generic line.

        Falls back to the keyword query on ANY failure — no embedding key, no
        pgvector, embeddings still backfilling, DB hiccup.
        """
        snippets = await self._vector_past_snippets(
            user_id, user_message, exclude_conversation_id, limit
        )
        if snippets is not None:
            return snippets
        return await self._keyword_past_snippets(
            user_id, user_message, exclude_conversation_id, limit
        )

    async def _vector_past_snippets(
        self,
        user_id: str,
        user_message: str,
        exclude_conversation_id: str,
        limit: int = 3,
    ) -> Optional[List[PastSnippet]]:
        """
        Cosine search over conversation_log.embedding (postgres + pgvector only).

        Returns None (not []) when the vector path could not run, so the caller
        can tell "unavailable, use keywords" apart from "ran, nothing cleared
        the floor".
        """
        if not user_message.strip():
            return None

        try:
            from sqlalchemy import text as sql_text
            from app.core.database import AsyncSessionLocal
            from app.services.embeddings import embed_one, to_pgvector

            vector = await embed_one(user_message)
            if vector is None:
                return None

            # 1 - cosine_distance is cosine similarity; pgvector's <=> is the distance.
            # Both the floor and the ordering use the boosted score, so a memory
            # just under the floor surfaces on the same terms it ranks by.
            stmt = sql_text("""
                WITH scored AS (
                    SELECT role, content, created_at,
                           1 - (embedding <=> CAST(:q AS vector))
                             + CASE WHEN role = 'memory' THEN :memory_boost ELSE 0 END AS score
                    FROM conversation_log
                    WHERE user_id = :user_id
                      AND conversation_id != :exclude_id
                      AND embedding IS NOT NULL
                      AND length(content) >= :min_chars
                )
                SELECT role, content, created_at, score FROM scored
                WHERE score >= :floor
                ORDER BY score DESC
                LIMIT :limit
            """)

            async with AsyncSessionLocal() as session:
                rows = (await session.execute(stmt, {
                    "q": to_pgvector(vector),
                    "user_id": str(user_id),
                    "exclude_id": str(exclude_conversation_id),
                    "min_chars": MIN_RECALL_CHARS,
                    "memory_boost": MEMORY_SIMILARITY_BOOST,
                    "floor": RECALL_SIMILARITY_FLOOR,
                    "limit": limit,
                })).all()

            logger.debug(
                "Vector recall: %d hit(s) over floor %.2f", len(rows), RECALL_SIMILARITY_FLOOR
            )
            return [
                PastSnippet(
                    role=row.role,
                    content=(
                        row.content[:_SNIPPET_CHARS].rstrip() + "..."
                        if len(row.content) > _SNIPPET_CHARS else row.content
                    ),
                    age=_humanize_age(row.created_at),
                )
                for row in rows
            ]

        except Exception as e:
            logger.warning(f"Vector recall unavailable, falling back to keywords: {e}")
            return None

    async def _keyword_past_snippets(
        self,
        user_id: str,
        user_message: str,
        exclude_conversation_id: str,
        limit: int = 3
    ) -> List[PastSnippet]:
        """
        Fallback recall: lines from OTHER conversations that share keywords with
        the incoming message, most-matching first, then most recent.

        Works on sqlite and on postgres without pgvector, which is exactly why it
        stays: it is what runs whenever the semantic path can't.

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