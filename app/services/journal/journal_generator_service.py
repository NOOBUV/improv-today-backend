"""
Journal Generation Service for Ava's Daily Journal Entries.
Handles LLM integration for creative writing with Fleabag-inspired voice.
"""

import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, date, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, and_
import openai
from openai import OpenAI

from app.core.config import settings
from app.models.simulation import GlobalEvents, AvaGlobalState
from app.models.journal import JournalEntries

logger = logging.getLogger(__name__)


class JournalGeneratorService:
    """Service for generating daily journal entries with LLM integration"""

    def __init__(self):
        """Initialize the journal generator with OpenAI client"""
        self.client = None
        if settings.openai_api_key and settings.openai_api_key != "":
            try:
                self.client = OpenAI(api_key=settings.openai_api_key)
                logger.info("OpenAI client initialized for journal generation")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
        else:
            logger.warning("No OpenAI API key found. Journal generation will use fallback content.")

    async def generate_daily_journal(self, session: AsyncSession, target_date: date = None) -> Optional[Dict[str, Any]]:
        """
        Generate a daily journal entry for the specified date.

        Args:
            session: Database session
            target_date: Date to generate journal for (defaults to today)

        Returns:
            Dictionary containing generated journal entry data
        """
        if target_date is None:
            target_date = date.today()

        try:
            # Aggregate daily events and emotional context
            daily_context = await self._aggregate_daily_context(session, target_date)

            if not daily_context["events"]:
                logger.info(f"No events found for {target_date}, skipping journal generation")
                return None

            # Generate journal content using LLM
            journal_content = await self._generate_journal_content(daily_context)

            # Prepare journal entry data
            journal_entry = {
                "entry_date": target_date,
                "content": journal_content,
                "status": "draft",
                "events_processed": len(daily_context["events"]),
                "emotional_theme": daily_context.get("dominant_emotion", "neutral"),
                "generated_at": datetime.now(timezone.utc)
            }

            logger.info(f"Generated journal entry for {target_date} with {len(daily_context['events'])} events")
            return journal_entry

        except Exception as e:
            logger.error(f"Error generating daily journal for {target_date}: {e}")
            return None

    async def _aggregate_daily_context(self, session: AsyncSession, target_date: date) -> Dict[str, Any]:
        """
        Aggregate events and emotional context for the specified date.

        Args:
            session: Database session
            target_date: Date to aggregate data for

        Returns:
            Dictionary containing daily context data
        """
        # Get all events for the target date
        start_datetime = datetime.combine(target_date, datetime.min.time())
        end_datetime = datetime.combine(target_date, datetime.max.time())

        events_result = await session.execute(
            select(GlobalEvents)
            .where(
                and_(
                    GlobalEvents.timestamp >= start_datetime,
                    GlobalEvents.timestamp <= end_datetime
                )
            )
            .order_by(desc(GlobalEvents.timestamp))
        )
        events = events_result.scalars().all()

        # Get current emotional state
        emotional_state = await self._get_emotional_state(session)

        # Analyze events for significance and emotional impact
        significant_events = self._rank_events_by_significance(events)

        context = {
            "target_date": target_date,
            "events": [self._format_event_for_context(event) for event in significant_events],
            "emotional_state": emotional_state,
            "dominant_emotion": self._determine_dominant_emotion(events, emotional_state),
            "event_count": len(events),
            "emotional_arc": self._trace_emotional_arc(events)
        }

        return context

    async def _get_emotional_state(self, session: AsyncSession) -> Dict[str, Any]:
        """Get current emotional state from AvaGlobalState"""
        state_result = await session.execute(
            select(AvaGlobalState)
            .where(
                AvaGlobalState.trait_name.in_(["mood", "stress", "energy"])
            )
        )
        states = state_result.scalars().all()

        emotional_state = {}
        for state in states:
            emotional_state[state.trait_name] = {
                "value": state.value,
                "numeric_value": state.numeric_value,
                "trend": state.trend
            }

        return emotional_state

    def _rank_events_by_significance(self, events: List[GlobalEvents]) -> List[GlobalEvents]:
        """Rank events by significance for journal content"""
        def significance_score(event):
            score = 0

            # Intensity factor
            if event.intensity:
                score += event.intensity * 2

            # Emotional impact factor
            if event.impact_mood == "positive":
                score += 3
            elif event.impact_mood == "negative":
                score += 4  # Negative events often more journalworthy

            # Event type factor
            event_type_weights = {
                "work": 2,
                "social": 3,
                "personal": 4
            }
            score += event_type_weights.get(event.event_type, 1)

            # Has emotional reaction (processed by consciousness)
            if event.emotional_reaction:
                score += 5

            return score

        sorted_events = sorted(events, key=significance_score, reverse=True)
        return sorted_events[:5]  # Top 5 most significant events

    def _format_event_for_context(self, event: GlobalEvents) -> Dict[str, Any]:
        """Format event data for LLM context"""
        return {
            "type": event.event_type,
            "summary": event.summary,
            "time": event.timestamp.strftime("%H:%M"),
            "intensity": event.intensity,
            "mood_impact": event.impact_mood,
            "emotional_reaction": event.emotional_reaction,
            "chosen_action": event.chosen_action,
            "internal_thoughts": event.internal_thoughts
        }

    def _determine_dominant_emotion(self, events: List[GlobalEvents], emotional_state: Dict) -> str:
        """Determine the dominant emotional theme of the day"""
        if not events:
            return "neutral"

        # Count emotional impacts
        emotion_counts = {}
        for event in events:
            if event.impact_mood:
                emotion_counts[event.impact_mood] = emotion_counts.get(event.impact_mood, 0) + 1

        if not emotion_counts:
            return "neutral"

        return max(emotion_counts.items(), key=lambda x: x[1])[0]

    def _trace_emotional_arc(self, events: List[GlobalEvents]) -> str:
        """Trace the emotional arc throughout the day"""
        if len(events) < 2:
            return "stable"

        # Simple arc detection based on event progression
        mood_progression = []
        for event in sorted(events, key=lambda e: e.timestamp):
            if event.impact_mood == "positive":
                mood_progression.append(1)
            elif event.impact_mood == "negative":
                mood_progression.append(-1)
            else:
                mood_progression.append(0)

        if not mood_progression:
            return "stable"

        start_mood = sum(mood_progression[:len(mood_progression)//3])
        end_mood = sum(mood_progression[-len(mood_progression)//3:])

        if end_mood > start_mood:
            return "improving"
        elif end_mood < start_mood:
            return "declining"
        else:
            return "stable"

    async def _generate_journal_content(self, context: Dict[str, Any]) -> str:
        """Generate journal content using LLM with Fleabag-inspired voice"""
        if not self.client:
            return self._generate_fallback_content(context)

        try:
            prompt = self._build_journal_prompt(context)

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": self._get_character_voice_instructions()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=300,
                temperature=0.8,
                presence_penalty=0.2,
                frequency_penalty=0.1
            )

            content = response.choices[0].message.content.strip()
            return content

        except Exception as e:
            logger.error(f"LLM journal generation failed: {e}")
            return self._generate_fallback_content(context)

    def _get_character_voice_instructions(self) -> str:
        """Character voice instructions for Fleabag-inspired writing"""
        return """You are Ava, writing in your personal journal. Your voice is:

- Witty and conversational, like Fleabag
- Self-aware and slightly cynical but ultimately vulnerable
- Uses humor to deflect from deeper emotions
- Breaks the fourth wall occasionally (talking to the reader)
- Honest about both victories and failures
- Uses modern, casual language
- Not overly dramatic but emotionally genuine
- Observant about human behavior and social dynamics

Write as if you're talking to a close friend or directly to the reader. Be authentic, funny, and real. Keep it under 280 characters (tweet-length) for social media compatibility."""

    def _build_journal_prompt(self, context: Dict[str, Any]) -> str:
        """Build the creative writing prompt for journal generation"""
        events_summary = ""
        if context["events"]:
            events_summary = "\n".join([
                f"- {event['time']}: {event['summary']}" +
                (f" (felt: {event['emotional_reaction']})" if event['emotional_reaction'] else "")
                for event in context["events"][:3]  # Top 3 events
            ])

        emotional_context = ""
        if context["emotional_state"]:
            mood_info = context["emotional_state"].get("mood", {})
            if mood_info:
                emotional_context = f"Current mood: {mood_info.get('value', 'neutral')}"

        prompt = f"""Here's what happened in your day on {context['target_date'].strftime('%B %d, %Y')}:

{events_summary}

{emotional_context}
Overall emotional theme: {context['dominant_emotion']}
Day's emotional arc: {context['emotional_arc']}

Write a short, witty journal entry reflecting on the most significant moment from today. Be authentic, use humor, and capture your genuine reaction to what happened. Remember - this might end up on social media, so keep it engaging but personal."""

        return prompt

    def _generate_fallback_content(self, context: Dict[str, Any]) -> str:
        """Generate fallback content when LLM is unavailable"""
        target_date = context["target_date"].strftime("%B %d, %Y")
        event_count = context["event_count"]
        dominant_emotion = context["dominant_emotion"]

        fallback_templates = [
            f"Well, {target_date} happened. {event_count} things occurred, most of them {dominant_emotion}. The usual chaos, really. At least I'm still here to write about it. 🤷‍♀️",
            f"Today's emotional forecast: {dominant_emotion} with a chance of existential dread. {event_count} events survived. Tomorrow will probably be similar. Such is life. ✨",
            f"{target_date}: Another day in paradise (and by paradise, I mean my beautifully chaotic life). {event_count} moments worth remembering, all very {dominant_emotion}. #authentic"
        ]

        import random
        return random.choice(fallback_templates)