"""
Event Selection Service - Uses existing simulation global_events with Redis-based tracking
to prevent Clara from repeating the same life events to the same user.
"""

import logging
from typing import List, Dict, Any, Optional
from app.services.event_usage_tracker import EventUsageTracker
from app.services.simulation.state_manager import StateManagerService

logger = logging.getLogger(__name__)


class EventSelectionService:
    """
    Service that selects fresh events from Clara's existing simulation system
    while preventing repetition using Redis-based tracking.
    """

    def __init__(self):
        self.event_tracker = EventUsageTracker()
        self.state_manager = StateManagerService()

    async def get_fresh_events_for_conversation(
        self,
        user_id: str,
        conversation_id: str,
        max_events: int = 3,
        hours_back: int = 72,  # Get events from last 3 days
        avoid_recent_days: int = 7  # Don't repeat events used in last week
    ) -> List[Dict[str, Any]]:
        """
        Get fresh events from Clara's simulation that haven't been mentioned to this user recently.

        Args:
            user_id: User identifier
            conversation_id: Conversation session ID
            max_events: Maximum number of events to return
            hours_back: How far back to look for simulation events
            avoid_recent_days: Don't reuse events mentioned in last N days

        Returns:
            List of fresh events with tracking metadata
        """
        try:
            # Get recent events from Clara's simulation
            all_recent_events = await self.state_manager.get_recent_events(
                hours_back=hours_back,
                max_count=max_events * 3  # Get more events than needed for selection
            )

            if not all_recent_events:
                logger.warning("No recent events available from simulation")
                return []

            # Convert simulation events to standardized format with unique IDs
            standardized_events = []
            for event in all_recent_events:
                event_id = event.get("event_id") or f"sim_{event.get('timestamp', '')}_{event.get('summary', '')[:20]}"
                standardized_event = {
                    "id": event_id,
                    "summary": event.get("summary", ""),
                    "event_type": event.get("event_type", "personal"),
                    "timestamp": event.get("timestamp"),
                    "intensity": event.get("intensity", 5),
                    "hours_ago": event.get("hours_ago", 0),
                    # Keep original simulation data
                    "original_event": event
                }
                standardized_events.append(standardized_event)

            # Use event tracker to get fresh events not recently mentioned to this user
            fresh_events = await self.event_tracker.get_fresh_events(
                user_id=user_id,
                event_pool=standardized_events,
                max_events=max_events,
                avoid_recent_days=avoid_recent_days
            )

            logger.info(f"Selected {len(fresh_events)} fresh events for user {user_id} from {len(all_recent_events)} available")
            return fresh_events

        except Exception as e:
            logger.error(f"Error getting fresh events for conversation: {e}")
            return []

    async def track_events_mentioned_in_response(
        self,
        user_id: str,
        conversation_id: str,
        events_mentioned: List[Dict[str, Any]]
    ) -> bool:
        """
        Track which events Clara mentioned in her response to prevent future repetition.

        Args:
            user_id: User identifier
            conversation_id: Conversation session ID
            events_mentioned: List of events that were referenced in the response

        Returns:
            Success status
        """
        try:
            event_ids = [event.get("id") for event in events_mentioned if event.get("id")]

            if not event_ids:
                return True

            success = await self.event_tracker.track_events_used(
                user_id=user_id,
                conversation_id=conversation_id,
                events_used=event_ids
            )

            if success:
                logger.debug(f"Tracked {len(event_ids)} events mentioned to user {user_id}")
            else:
                logger.warning(f"Failed to track events for user {user_id}")

            return success

        except Exception as e:
            logger.error(f"Error tracking mentioned events: {e}")
            return False

    async def get_contextual_events(
        self,
        user_id: str,
        conversation_id: str,
        user_message: str,
        max_events: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Get events that are contextually relevant to the user's message
        while avoiding repetition.

        Args:
            user_id: User identifier
            conversation_id: Conversation session ID
            user_message: User's message to find relevant events for
            max_events: Maximum number of events to return

        Returns:
            List of contextually relevant fresh events
        """
        try:
            # Get fresh events first
            fresh_events = await self.get_fresh_events_for_conversation(
                user_id=user_id,
                conversation_id=conversation_id,
                max_events=max_events * 2  # Get more for filtering
            )

            if not fresh_events:
                return []

            # Simple contextual matching
            user_lower = user_message.lower()
            contextual_events = []

            for event in fresh_events:
                relevance_score = 0
                event_text = f"{event.get('summary', '')} {event.get('event_type', '')}".lower()

                # Keyword matching for different contexts
                work_keywords = ["work", "job", "meeting", "project", "deadline", "colleague", "client", "office"]
                social_keywords = ["friend", "party", "dinner", "social", "people", "hang out", "meet", "event"]
                personal_keywords = ["personal", "home", "family", "self", "alone", "thinking", "feeling"]
                stress_keywords = ["stress", "busy", "overwhelmed", "tired", "pressure", "difficult", "hard"]

                # Score based on keyword matches
                if any(keyword in user_lower for keyword in work_keywords):
                    if any(keyword in event_text for keyword in work_keywords) or event.get("event_type") == "work":
                        relevance_score += 3

                if any(keyword in user_lower for keyword in social_keywords):
                    if any(keyword in event_text for keyword in social_keywords) or event.get("event_type") == "social":
                        relevance_score += 3

                if any(keyword in user_lower for keyword in personal_keywords):
                    if any(keyword in event_text for keyword in personal_keywords) or event.get("event_type") == "personal":
                        relevance_score += 3

                if any(keyword in user_lower for keyword in stress_keywords):
                    if event.get("intensity", 5) >= 6:  # High intensity events for stress topics
                        relevance_score += 2

                # Include event if it has any relevance
                if relevance_score > 0:
                    event_with_score = event.copy()
                    event_with_score["relevance_score"] = relevance_score
                    contextual_events.append(event_with_score)

            # Sort by relevance and return top matches
            contextual_events.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            result = contextual_events[:max_events]

            # If no contextual matches, return random fresh events
            if not result:
                result = fresh_events[:max_events]

            logger.debug(f"Selected {len(result)} contextual events for user message: {user_message[:50]}...")
            return result

        except Exception as e:
            logger.error(f"Error getting contextual events: {e}")
            return []

    async def get_event_usage_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Get summary of event usage for a specific user (debugging/monitoring).

        Args:
            user_id: User identifier

        Returns:
            Event usage summary for the user
        """
        try:
            user_history = await self.event_tracker.get_user_event_history(user_id)
            global_stats = await self.event_tracker.get_event_usage_stats()

            return {
                "user_history": user_history,
                "global_stats": global_stats,
                "summary": {
                    "user_events_count": user_history.get("total_events_used", 0),
                    "global_events_count": len(global_stats)
                }
            }

        except Exception as e:
            logger.error(f"Error getting event usage summary: {e}")
            return {"error": str(e)}

    async def cleanup_old_event_data(self, days_old: int = 30) -> Dict[str, Any]:
        """
        Cleanup old event tracking data.

        Args:
            days_old: Remove tracking data older than this many days

        Returns:
            Cleanup statistics
        """
        try:
            return await self.event_tracker.cleanup_old_data(days_old)
        except Exception as e:
            logger.error(f"Error during event data cleanup: {e}")
            return {"error": str(e)}