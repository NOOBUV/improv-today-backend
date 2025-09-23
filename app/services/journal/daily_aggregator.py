"""
Daily Event Aggregation Service.
Handles ranking and aggregation of events by significance for journal content.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date, time, timedelta, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, and_, func

from app.models.simulation import GlobalEvents, ClaraGlobalState

logger = logging.getLogger(__name__)


class DailyAggregatorService:
    """Service for aggregating and ranking daily events for journal generation"""

    def __init__(self):
        """Initialize the daily aggregator service"""
        self.significance_weights = {
            "intensity": 2.0,
            "emotional_impact": 3.0,
            "event_type": 1.5,
            "consciousness_response": 4.0,
            "recency": 1.0
        }

    async def aggregate_daily_events(
        self,
        session: AsyncSession,
        target_date: date,
        max_events: int = 5
    ) -> Dict[str, Any]:
        """
        Aggregate and rank events for the specified date.

        Args:
            session: Database session
            target_date: Date to aggregate events for
            max_events: Maximum number of events to include

        Returns:
            Dictionary containing aggregated event data and context
        """
        try:
            # Get all events for the target date
            events = await self._get_daily_events(session, target_date)

            if not events:
                logger.info(f"No events found for {target_date}")
                return self._empty_aggregation(target_date)

            # Rank events by significance
            ranked_events = self._rank_events_by_significance(events)

            # Get emotional context
            emotional_context = await self._get_emotional_context(session, target_date)

            # Analyze emotional arc throughout the day
            emotional_arc = self._analyze_emotional_arc(events)

            # Select most significant events
            significant_events = ranked_events[:max_events]

            # Build aggregation result
            aggregation = {
                "target_date": target_date,
                "total_events": len(events),
                "significant_events": [
                    self._format_event_for_aggregation(event, rank)
                    for rank, event in enumerate(significant_events, 1)
                ],
                "emotional_context": emotional_context,
                "emotional_arc": emotional_arc,
                "dominant_emotion": self._determine_dominant_emotion(events),
                "activity_summary": self._summarize_daily_activity(events),
                "significance_scores": [
                    self._calculate_significance_score(event) for event in significant_events
                ]
            }

            logger.info(
                f"Aggregated {len(significant_events)} significant events "
                f"from {len(events)} total events for {target_date}"
            )

            return aggregation

        except Exception as e:
            logger.error(f"Error aggregating daily events for {target_date}: {e}")
            return self._empty_aggregation(target_date)

    async def _get_daily_events(self, session: AsyncSession, target_date: date) -> List[GlobalEvents]:
        """Retrieve all events for the specified date"""
        start_datetime = datetime.combine(target_date, time.min)
        end_datetime = datetime.combine(target_date, time.max)

        result = await session.execute(
            select(GlobalEvents)
            .where(
                and_(
                    GlobalEvents.timestamp >= start_datetime,
                    GlobalEvents.timestamp <= end_datetime
                )
            )
            .order_by(GlobalEvents.timestamp)
        )

        return result.scalars().all()

    async def _get_emotional_context(self, session: AsyncSession, target_date: date) -> Dict[str, Any]:
        """Get emotional state context around the target date"""
        try:
            # Get current emotional state
            result = await session.execute(
                select(ClaraGlobalState)
                .where(
                    ClaraGlobalState.trait_name.in_(["mood", "stress", "energy", "confidence"])
                )
            )
            states = result.scalars().all()

            emotional_context = {}
            for state in states:
                emotional_context[state.trait_name] = {
                    "value": state.value,
                    "numeric_value": state.numeric_value,
                    "trend": state.trend,
                    "last_updated": state.last_updated
                }

            return emotional_context

        except Exception as e:
            logger.error(f"Error getting emotional context: {e}")
            return {}

    def _rank_events_by_significance(self, events: List[GlobalEvents]) -> List[GlobalEvents]:
        """Rank events by significance using weighted scoring"""
        scored_events = []

        for event in events:
            score = self._calculate_significance_score(event)
            scored_events.append((score, event))

        # Sort by score (descending)
        scored_events.sort(key=lambda x: x[0], reverse=True)

        return [event for score, event in scored_events]

    def _calculate_significance_score(self, event: GlobalEvents) -> float:
        """Calculate significance score for an event"""
        score = 0.0

        # Intensity factor
        if event.intensity:
            score += event.intensity * self.significance_weights["intensity"]

        # Emotional impact factor
        if event.impact_mood:
            impact_score = {
                "positive": 3.0,
                "negative": 4.0,  # Negative events often more journalworthy
                "neutral": 1.0
            }.get(event.impact_mood, 1.0)
            score += impact_score * self.significance_weights["emotional_impact"]

        # Event type factor
        type_score = {
            "personal": 4.0,  # Personal events most significant
            "social": 3.0,
            "work": 2.0,
            "routine": 1.0
        }.get(event.event_type, 2.0)
        score += type_score * self.significance_weights["event_type"]

        # Consciousness response factor (has emotional reaction)
        if event.emotional_reaction or event.internal_thoughts:
            score += 5.0 * self.significance_weights["consciousness_response"]

        # Recency factor (more recent events slightly favored)
        hours_old = (datetime.now(timezone.utc) - event.timestamp).total_seconds() / 3600
        recency_score = max(0, 24 - hours_old) / 24  # Score 0-1 based on how recent
        score += recency_score * self.significance_weights["recency"]

        return score

    def _analyze_emotional_arc(self, events: List[GlobalEvents]) -> str:
        """Analyze the emotional arc throughout the day"""
        if len(events) < 2:
            return "stable"

        # Track mood progression through the day
        mood_scores = []
        for event in sorted(events, key=lambda e: e.timestamp):
            if event.impact_mood == "positive":
                mood_scores.append(1)
            elif event.impact_mood == "negative":
                mood_scores.append(-1)
            else:
                mood_scores.append(0)

        if not mood_scores:
            return "stable"

        # Simple trend analysis
        first_half = mood_scores[:len(mood_scores)//2] if len(mood_scores) > 2 else mood_scores[:1]
        second_half = mood_scores[len(mood_scores)//2:] if len(mood_scores) > 2 else mood_scores[-1:]

        first_avg = sum(first_half) / len(first_half) if first_half else 0
        second_avg = sum(second_half) / len(second_half) if second_half else 0

        if second_avg > first_avg + 0.3:
            return "improving"
        elif second_avg < first_avg - 0.3:
            return "declining"
        elif abs(second_avg - first_avg) < 0.1:
            return "stable"
        else:
            return "mixed"

    def _determine_dominant_emotion(self, events: List[GlobalEvents]) -> str:
        """Determine the dominant emotional theme of the day"""
        if not events:
            return "neutral"

        emotion_counts = {}
        emotion_intensity = {}

        for event in events:
            if event.impact_mood:
                emotion = event.impact_mood
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

                # Weight by intensity
                intensity = event.intensity or 5
                emotion_intensity[emotion] = emotion_intensity.get(emotion, 0) + intensity

        if not emotion_counts:
            return "neutral"

        # Determine dominant emotion by weighted score
        weighted_scores = {}
        for emotion in emotion_counts:
            count_score = emotion_counts[emotion] * 2
            intensity_score = emotion_intensity.get(emotion, 0)
            weighted_scores[emotion] = count_score + intensity_score

        dominant = max(weighted_scores.items(), key=lambda x: x[1])[0]

        # Check if it's truly dominant or mixed
        total_score = sum(weighted_scores.values())
        dominant_percentage = weighted_scores[dominant] / total_score if total_score > 0 else 0

        if dominant_percentage < 0.6:  # Less than 60% dominance
            return "mixed"

        return dominant

    def _summarize_daily_activity(self, events: List[GlobalEvents]) -> Dict[str, Any]:
        """Summarize the day's activity patterns"""
        if not events:
            return {"activity_level": "low", "variety": "low", "peak_time": "unknown"}

        # Analyze activity distribution
        event_types = {}
        hourly_distribution = {}

        for event in events:
            # Event type distribution
            event_type = event.event_type
            event_types[event_type] = event_types.get(event_type, 0) + 1

            # Hourly distribution
            hour = event.timestamp.hour
            hourly_distribution[hour] = hourly_distribution.get(hour, 0) + 1

        # Determine activity level
        activity_level = "high" if len(events) > 8 else "medium" if len(events) > 4 else "low"

        # Determine variety
        variety = "high" if len(event_types) >= 3 else "medium" if len(event_types) > 1 else "low"

        # Find peak activity time
        peak_hour = max(hourly_distribution.items(), key=lambda x: x[1])[0] if hourly_distribution else 12
        peak_time = f"{peak_hour:02d}:00"

        return {
            "activity_level": activity_level,
            "variety": variety,
            "peak_time": peak_time,
            "event_types": event_types,
            "total_events": len(events)
        }

    def _format_event_for_aggregation(self, event: GlobalEvents, rank: int) -> Dict[str, Any]:
        """Format event data for aggregation output"""
        return {
            "rank": rank,
            "event_id": event.event_id,
            "timestamp": event.timestamp.isoformat(),
            "time": event.timestamp.strftime("%H:%M"),
            "type": event.event_type,
            "summary": event.summary,
            "intensity": event.intensity,
            "mood_impact": event.impact_mood,
            "energy_impact": event.impact_energy,
            "stress_impact": event.impact_stress,
            "emotional_reaction": event.emotional_reaction,
            "chosen_action": event.chosen_action,
            "internal_thoughts": event.internal_thoughts,
            "significance_score": self._calculate_significance_score(event)
        }

    def _empty_aggregation(self, target_date: date) -> Dict[str, Any]:
        """Return empty aggregation structure for days with no events"""
        return {
            "target_date": target_date,
            "total_events": 0,
            "significant_events": [],
            "emotional_context": {},
            "emotional_arc": "stable",
            "dominant_emotion": "neutral",
            "activity_summary": {
                "activity_level": "low",
                "variety": "low",
                "peak_time": "unknown",
                "event_types": {},
                "total_events": 0
            },
            "significance_scores": []
        }

    async def get_event_significance_stats(self, session: AsyncSession, days: int = 7) -> Dict[str, Any]:
        """Get statistics about event significance over the last N days"""
        try:
            end_date = date.today()
            start_date = end_date - timedelta(days=days)

            start_datetime = datetime.combine(start_date, time.min)
            end_datetime = datetime.combine(end_date, time.max)

            result = await session.execute(
                select(GlobalEvents)
                .where(
                    and_(
                        GlobalEvents.timestamp >= start_datetime,
                        GlobalEvents.timestamp <= end_datetime
                    )
                )
            )
            events = result.scalars().all()

            if not events:
                return {"total_events": 0, "avg_significance": 0, "high_significance_events": 0}

            scores = [self._calculate_significance_score(event) for event in events]
            high_significance_count = len([s for s in scores if s > 15])  # Threshold for "high significance"

            return {
                "total_events": len(events),
                "avg_significance": sum(scores) / len(scores),
                "high_significance_events": high_significance_count,
                "significance_distribution": {
                    "low": len([s for s in scores if s < 5]),
                    "medium": len([s for s in scores if 5 <= s < 15]),
                    "high": high_significance_count
                }
            }

        except Exception as e:
            logger.error(f"Error getting significance stats: {e}")
            return {"error": str(e)}