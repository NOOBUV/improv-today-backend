"""
Dynamic Content Selector with event freshness weighting and relevance scoring.
Intelligently selects and prioritizes content based on question routing and event analysis.
"""

import logging
from typing import Dict, List, Any, Set
from datetime import datetime, timezone
from dataclasses import dataclass
import re

from app.services.content_source_router import ContentSourceRouter, ContentRoutingResult, QuestionCategory

logger = logging.getLogger(__name__)


@dataclass
class EventScore:
    """Scoring information for a simulation event."""
    event_id: str
    summary: str
    relevance_score: float
    freshness_score: float
    total_score: float
    reasoning: str


@dataclass
class ContentSelection:
    """Result of dynamic content selection."""
    selected_events: List[Dict[str, Any]]
    event_scores: List[EventScore]
    entities_found: Dict[str, List[str]]
    content_strategy: str
    total_events_analyzed: int


class DynamicContentSelector:
    """Dynamically selects and scores content based on question analysis and event freshness."""

    def __init__(self):
        self.router = ContentSourceRouter()

    def select_content(
        self,
        user_message: str,
        recent_events: List[Dict[str, Any]],
        max_events: int = 3
    ) -> ContentSelection:
        """
        Select and score content dynamically based on user question and available events.

        Args:
            user_message: User's question/message
            recent_events: List of recent simulation events
            max_events: Maximum number of events to select

        Returns:
            ContentSelection with prioritized events and metadata
        """
        try:
            # Extract entities from recent events for dynamic routing
            entities = self._extract_entities_from_events(recent_events)

            # Route the question to determine content strategy
            routing_result = self.router.analyze_question(user_message, entities)

            logger.debug(f"Question routing: {routing_result.question_category.value} "
                        f"(conf: {routing_result.confidence:.2f})")

            # Score all events for relevance and freshness
            event_scores = self._score_events(
                user_message,
                recent_events,
                routing_result,
                entities
            )

            # Select top events based on scores
            selected_scores = sorted(event_scores, key=lambda x: x.total_score, reverse=True)[:max_events]
            selected_events = [
                next(event for event in recent_events if event.get('event_id') == score.event_id)
                for score in selected_scores
            ]

            # Get content strategy description
            content_strategy = self._get_content_strategy_description(routing_result, selected_scores)

            return ContentSelection(
                selected_events=selected_events,
                event_scores=selected_scores,
                entities_found=entities,
                content_strategy=content_strategy,
                total_events_analyzed=len(recent_events)
            )

        except Exception as e:
            logger.error(f"Error in dynamic content selection: {e}")
            # Fallback: return most recent events
            return ContentSelection(
                selected_events=recent_events[:max_events] if recent_events else [],
                event_scores=[],
                entities_found={},
                content_strategy="Fallback: Using most recent events",
                total_events_analyzed=len(recent_events)
            )

    def _extract_entities_from_events(self, events: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Extract people names and event types from recent events."""
        entities = {"people": set(), "events": set()}

        for event in events:
            summary = event.get('summary', '')
            event_type = event.get('event_type', '')

            # Extract potential person names (capitalized words in summaries)
            # Look for patterns like "with [Name]", "meeting with [Name]", "[Name] called"
            name_patterns = [
                r'\bwith\s+([A-Z][a-z]+)',
                r'\b([A-Z][a-z]+)\s+(?:called|texted|visited)',
                r'\b(?:meeting|dinner|coffee|lunch)\s+with\s+([A-Z][a-z]+)',
                r'\b([A-Z][a-z]+)\s+(?:and I|came over|stopped by)'
            ]

            for pattern in name_patterns:
                matches = re.findall(pattern, summary)
                for match in matches:
                    if len(match) > 2:  # Filter out short words like "I", "A"
                        entities["people"].add(match)

            # Extract event types and activities
            activity_patterns = [
                r'\b(meeting|call|session|review)\b',
                r'\b(dinner|lunch|coffee|drinks|breakfast)\b',
                r'\b(workout|gym|exercise|running)\b',
                r'\b(shopping|errands|cleaning|organizing)\b',
                r'\b(reading|journaling|writing)\b',
                r'\b(networking|social|party|gathering)\b'
            ]

            for pattern in activity_patterns:
                matches = re.findall(pattern, summary.lower())
                entities["events"].update(matches)

            # Add event type as entity
            if event_type:
                entities["events"].add(event_type)

        # Convert sets to lists and filter
        return {
            "people": list(entities["people"]),
            "events": list(entities["events"])
        }

    def _score_events(
        self,
        user_message: str,
        events: List[Dict[str, Any]],
        routing_result: ContentRoutingResult,
        entities: Dict[str, List[str]]
    ) -> List[EventScore]:
        """Score events based on relevance and freshness."""
        event_scores = []
        message_lower = user_message.lower()

        for event in events:
            try:
                # Calculate relevance score
                relevance_score = self._calculate_relevance_score(
                    event, message_lower, routing_result, entities
                )

                # Calculate freshness score
                freshness_score = self._calculate_freshness_score(event, routing_result.question_category)

                # Calculate total weighted score
                total_score = self._calculate_total_score(
                    relevance_score, freshness_score, routing_result
                )

                reasoning = self._generate_score_reasoning(
                    event, relevance_score, freshness_score, routing_result
                )

                event_scores.append(EventScore(
                    event_id=event.get('event_id', ''),
                    summary=event.get('summary', ''),
                    relevance_score=relevance_score,
                    freshness_score=freshness_score,
                    total_score=total_score,
                    reasoning=reasoning
                ))

            except Exception as e:
                logger.warning(f"Error scoring event {event.get('event_id')}: {e}")
                continue

        return event_scores

    def _calculate_relevance_score(
        self,
        event: Dict[str, Any],
        message_lower: str,
        routing_result: ContentRoutingResult,
        entities: Dict[str, List[str]]
    ) -> float:
        """Calculate how relevant an event is to the user's question."""
        score = 0.0
        summary_lower = event.get('summary', '').lower()

        # Keyword matching from routing analysis
        for keyword in routing_result.keywords_found:
            if keyword.lower() in summary_lower:
                score += 0.3

        # Direct entity matches
        for person in entities.get('people', []):
            if person.lower() in message_lower and person.lower() in summary_lower:
                score += 0.5  # High score for person matches

        for event_type in entities.get('events', []):
            if event_type.lower() in message_lower and event_type.lower() in summary_lower:
                score += 0.4  # High score for event type matches

        # Question category specific bonuses
        if routing_result.question_category == QuestionCategory.SPECIFIC_PERSON:
            # Boost social events when asking about people
            if event.get('event_type') == 'social':
                score += 0.3

        elif routing_result.question_category == QuestionCategory.CURRENT_DAY:
            # Boost more recent and higher intensity events for "how's your day"
            intensity = event.get('intensity', 0)
            score += (intensity / 10) * 0.2

        elif routing_result.question_category == QuestionCategory.RECENT_LIFE:
            # Boost interesting/varied events for "how's life" questions
            if event.get('intensity', 0) >= 5:  # Interesting events
                score += 0.2

        # Intensity and impact bonuses
        intensity = event.get('intensity', 0)
        if intensity >= 7:  # High intensity events are usually more notable
            score += 0.2

        return min(score, 1.0)  # Cap at 1.0

    def _calculate_freshness_score(self, event: Dict[str, Any], category: QuestionCategory) -> float:
        """Calculate freshness score based on event age and question category."""
        try:
            hours_ago = event.get('hours_ago', 0)

            # Different freshness curves for different question types
            if category == QuestionCategory.CURRENT_DAY:
                # Heavily favor very recent events for "how's your day"
                if hours_ago <= 6:
                    return 1.0
                elif hours_ago <= 12:
                    return 0.8
                elif hours_ago <= 24:
                    return 0.5
                else:
                    return 0.1

            elif category in [QuestionCategory.RECENT_LIFE, QuestionCategory.SPECIFIC_PERSON]:
                # More gradual decay for "how's life" or person questions
                if hours_ago <= 12:
                    return 1.0
                elif hours_ago <= 24:
                    return 0.9
                elif hours_ago <= 48:
                    return 0.7
                elif hours_ago <= 72:
                    return 0.5
                else:
                    return 0.2

            else:
                # Default freshness curve
                if hours_ago <= 24:
                    return 1.0 - (hours_ago / 24) * 0.3
                elif hours_ago <= 72:
                    return 0.7 - ((hours_ago - 24) / 48) * 0.5
                else:
                    return 0.2

        except Exception:
            return 0.5  # Default fallback

    def _calculate_total_score(
        self,
        relevance_score: float,
        freshness_score: float,
        routing_result: ContentRoutingResult
    ) -> float:
        """Calculate weighted total score based on question category."""
        # Different weight distributions based on question type
        if routing_result.question_category == QuestionCategory.CURRENT_DAY:
            # For "how's your day" - freshness is very important
            return (relevance_score * 0.3) + (freshness_score * 0.7)

        elif routing_result.question_category == QuestionCategory.SPECIFIC_PERSON:
            # For person questions - relevance is more important than freshness
            return (relevance_score * 0.8) + (freshness_score * 0.2)

        elif routing_result.question_category == QuestionCategory.RECENT_LIFE:
            # For "how's life" - balanced approach
            return (relevance_score * 0.5) + (freshness_score * 0.5)

        else:
            # Default balanced scoring
            return (relevance_score * 0.6) + (freshness_score * 0.4)

    def _generate_score_reasoning(
        self,
        event: Dict[str, Any],
        relevance_score: float,
        freshness_score: float,
        routing_result: ContentRoutingResult
    ) -> str:
        """Generate human-readable reasoning for the scoring."""
        parts = []

        if relevance_score > 0.5:
            parts.append("High relevance to question")
        elif relevance_score > 0.2:
            parts.append("Moderate relevance")
        else:
            parts.append("Low relevance")

        hours_ago = event.get('hours_ago', 0)
        if hours_ago <= 12:
            parts.append("very recent")
        elif hours_ago <= 24:
            parts.append("recent")
        elif hours_ago <= 48:
            parts.append("somewhat recent")
        else:
            parts.append("older event")

        intensity = event.get('intensity', 0)
        if intensity >= 7:
            parts.append("high intensity")
        elif intensity >= 4:
            parts.append("moderate intensity")

        return f"{routing_result.question_category.value} question: {', '.join(parts)}"

    def _get_content_strategy_description(
        self,
        routing_result: ContentRoutingResult,
        selected_scores: List[EventScore]
    ) -> str:
        """Generate description of the content selection strategy used."""
        if not selected_scores:
            return f"No relevant events found for {routing_result.question_category.value} question"

        avg_relevance = sum(score.relevance_score for score in selected_scores) / len(selected_scores)
        avg_freshness = sum(score.freshness_score for score in selected_scores) / len(selected_scores)

        return (f"{routing_result.question_category.value} strategy: "
                f"Selected {len(selected_scores)} events with "
                f"avg relevance {avg_relevance:.2f}, avg freshness {avg_freshness:.2f}")