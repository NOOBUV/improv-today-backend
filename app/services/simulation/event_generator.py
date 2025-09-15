"""
Core event generation service for the simulation engine.
Implements predefined patterns with randomization for variety.
"""

import random
import logging
from datetime import datetime, time
from typing import List, Dict, Any, Optional
from celery import shared_task
import asyncio

from app.core.database import SessionLocal
from app.schemas.simulation_schemas import GlobalEventCreate, EventType, MoodImpact, ImpactLevel
from app.services.simulation.repository import SimulationRepository
from app.services.simulation.event_patterns import EventPatterns

logger = logging.getLogger(__name__)


class EventGeneratorService:
    """Service for generating daily life events for Ava."""

    def __init__(self):
        self.event_patterns = EventPatterns()

    async def generate_hourly_event(self, current_hour: int) -> Optional[GlobalEventCreate]:
        """
        Generate an event based on the current hour and predefined patterns.
        Returns None if no event should be generated for this hour.
        """
        try:
            # Get time-based event probability
            event_chance = self._get_hourly_event_chance(current_hour)

            # Random chance to generate an event
            if random.random() > event_chance:
                logger.debug(f"No event generated for hour {current_hour} (chance: {event_chance})")
                return None

            # Determine event type based on hour
            event_type = self._determine_event_type_by_hour(current_hour)

            # Generate event based on type
            event_data = await self._generate_event_by_type(event_type, current_hour)

            logger.info(f"Generated {event_type} event for hour {current_hour}: {event_data.summary[:50]}...")
            return event_data

        except Exception as e:
            logger.error(f"Error generating hourly event for hour {current_hour}: {e}")
            return None

    def _get_hourly_event_chance(self, hour: int) -> float:
        """Get the probability of generating an event for a specific hour."""
        # Higher probability during active hours, lower during sleep/quiet hours
        hour_probabilities = {
            0: 0.05,   # Midnight - very low
            1: 0.02,   # 1 AM - almost none
            2: 0.02,   # 2 AM - almost none
            3: 0.02,   # 3 AM - almost none
            4: 0.02,   # 4 AM - almost none
            5: 0.02,   # 5 AM - almost none
            6: 0.05,   # 6 AM - very low
            7: 0.15,   # 7 AM - morning routine
            8: 0.25,   # 8 AM - work start
            9: 0.30,   # 9 AM - active work
            10: 0.35,  # 10 AM - peak morning
            11: 0.30,  # 11 AM - active
            12: 0.40,  # Noon - lunch/social
            13: 0.35,  # 1 PM - post-lunch
            14: 0.30,  # 2 PM - afternoon work
            15: 0.35,  # 3 PM - active afternoon
            16: 0.30,  # 4 PM - late afternoon
            17: 0.40,  # 5 PM - end of work/social
            18: 0.45,  # 6 PM - dinner/evening social
            19: 0.40,  # 7 PM - evening activities
            20: 0.35,  # 8 PM - evening wind down
            21: 0.25,  # 9 PM - relaxation
            22: 0.15,  # 10 PM - preparing for bed
            23: 0.10,  # 11 PM - late evening
        }
        return hour_probabilities.get(hour, 0.20)  # Default 20% chance

    def _determine_event_type_by_hour(self, hour: int) -> EventType:
        """Determine the most likely event type based on the hour."""
        # Work hours (9-17) favor work events
        if 9 <= hour <= 17:
            weights = [0.60, 0.25, 0.15]  # work, social, personal
        # Evening hours (18-22) favor social events
        elif 18 <= hour <= 22:
            weights = [0.10, 0.60, 0.30]  # work, social, personal
        # Morning hours (6-8) and late evening (23-24) favor personal events
        elif hour in [6, 7, 8, 23, 0]:
            weights = [0.10, 0.20, 0.70]  # work, social, personal
        # Night hours favor personal events (if any)
        else:
            weights = [0.05, 0.15, 0.80]  # work, social, personal

        event_types = [EventType.WORK, EventType.SOCIAL, EventType.PERSONAL]
        return random.choices(event_types, weights=weights)[0]

    async def _generate_event_by_type(self, event_type: EventType, hour: int) -> GlobalEventCreate:
        """Generate a specific event based on type and hour."""
        if event_type == EventType.WORK:
            return self._generate_work_event(hour)
        elif event_type == EventType.SOCIAL:
            return self._generate_social_event(hour)
        else:  # PERSONAL
            return self._generate_personal_event(hour)

    def _generate_work_event(self, hour: int) -> GlobalEventCreate:
        """Generate a work-related event."""
        work_events = self.event_patterns.get_work_events_by_hour(hour)
        event_template = random.choice(work_events)

        # Add randomization to the template
        summary = self._randomize_event_text(event_template["summary"])
        intensity = event_template.get("intensity", random.randint(3, 7))

        return GlobalEventCreate(
            event_type=EventType.WORK,
            summary=summary,
            intensity=intensity,
            impact_mood=MoodImpact(event_template.get("mood_impact", "neutral")),
            impact_energy=ImpactLevel(event_template.get("energy_impact", "neutral")),
            impact_stress=ImpactLevel(event_template.get("stress_impact", "neutral"))
        )

    def _generate_social_event(self, hour: int) -> GlobalEventCreate:
        """Generate a social event."""
        social_events = self.event_patterns.get_social_events_by_hour(hour)
        event_template = random.choice(social_events)

        summary = self._randomize_event_text(event_template["summary"])
        intensity = event_template.get("intensity", random.randint(4, 8))

        return GlobalEventCreate(
            event_type=EventType.SOCIAL,
            summary=summary,
            intensity=intensity,
            impact_mood=MoodImpact(event_template.get("mood_impact", "positive")),
            impact_energy=ImpactLevel(event_template.get("energy_impact", "neutral")),
            impact_stress=ImpactLevel(event_template.get("stress_impact", "neutral"))
        )

    def _generate_personal_event(self, hour: int) -> GlobalEventCreate:
        """Generate a personal event."""
        personal_events = self.event_patterns.get_personal_events_by_hour(hour)
        event_template = random.choice(personal_events)

        summary = self._randomize_event_text(event_template["summary"])
        intensity = event_template.get("intensity", random.randint(2, 6))

        return GlobalEventCreate(
            event_type=EventType.PERSONAL,
            summary=summary,
            intensity=intensity,
            impact_mood=MoodImpact(event_template.get("mood_impact", "neutral")),
            impact_energy=ImpactLevel(event_template.get("energy_impact", "neutral")),
            impact_stress=ImpactLevel(event_template.get("stress_impact", "neutral"))
        )

    def _randomize_event_text(self, template: str) -> str:
        """Add randomization to event text templates."""
        # Replace placeholders with random variations
        variations = {
            "{colleague}": ["Sarah", "Mike", "Jessica", "David", "Emily", "Alex"],
            "{project}": ["quarterly report", "client presentation", "budget review", "team meeting", "project update"],
            "{friend}": ["Emma", "Jake", "Lisa", "Ryan", "Sophie", "Chris"],
            "{activity}": ["coffee", "lunch", "shopping", "movie", "walk", "chat"],
            "{meal}": ["breakfast", "lunch", "dinner", "snack"],
            "{exercise}": ["yoga", "running", "gym workout", "stretching", "walk"],
            "{hobby}": ["reading", "painting", "music", "writing", "cooking", "gardening"],
        }

        result = template
        for placeholder, options in variations.items():
            if placeholder in result:
                result = result.replace(placeholder, random.choice(options))

        return result


# Celery tasks for the simulation engine
@shared_task(bind=True, name="app.services.simulation.event_generator.generate_daily_event")
def generate_daily_event(self):
    """
    Celery task to generate a daily event.
    Runs every hour as scheduled by Celery Beat.
    """
    try:
        current_hour = datetime.now().hour
        logger.info(f"Starting event generation for hour {current_hour}")

        # Sync database operations for Celery task
        generator = EventGeneratorService()

        # Create event loop for async event generation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            event_data = loop.run_until_complete(generator.generate_hourly_event(current_hour))
        finally:
            loop.close()

        if event_data:
            # Save to database using sync session
            with SessionLocal() as db_session:
                try:
                    from app.models.simulation import GlobalEvents

                    db_event = GlobalEvents(
                        event_type=event_data.event_type,
                        summary=event_data.summary,
                        intensity=event_data.intensity,
                        impact_mood=event_data.impact_mood,
                        impact_energy=event_data.impact_energy,
                        impact_stress=event_data.impact_stress
                    )

                    db_session.add(db_event)
                    db_session.commit()
                    db_session.refresh(db_event)

                    logger.info(f"Created event {db_event.event_id}: {db_event.summary}")
                    return {
                        "success": True,
                        "event_id": db_event.event_id,
                        "event_type": db_event.event_type,
                        "summary": db_event.summary
                    }
                except Exception as e:
                    db_session.rollback()
                    logger.error(f"Error saving event to database: {e}")
                    raise
        else:
            logger.info(f"No event generated for hour {current_hour}")
            return {
                "success": True,
                "event_id": None,
                "message": f"No event generated for hour {current_hour}"
            }

    except Exception as e:
        logger.error(f"Error in generate_daily_event task: {e}")
        self.retry(countdown=300, max_retries=3)  # Retry after 5 minutes, max 3 times


@shared_task(bind=True, name="app.services.simulation.event_generator.process_pending_events")
def process_pending_events(self):
    """
    Celery task to process pending events.
    Runs every 15 minutes to ensure events are processed.
    """
    try:
        logger.info("Starting processing of pending events")

        # Sync database operations for Celery task
        with SessionLocal() as db_session:
            try:
                from app.models.simulation import GlobalEvents
                from app.schemas.simulation_schemas import EventStatus
                from sqlalchemy import select

                # Get unprocessed events
                pending_events = db_session.execute(
                    select(GlobalEvents)
                    .where(GlobalEvents.status == EventStatus.UNPROCESSED)
                    .order_by(GlobalEvents.timestamp)
                    .limit(10)
                ).scalars().all()

                processed_count = 0
                for event in pending_events:
                    # Mark as processed
                    event.status = EventStatus.PROCESSED
                    event.processed_at = datetime.utcnow()
                    processed_count += 1
                    logger.debug(f"Processed event {event.event_id}")

                db_session.commit()
                logger.info(f"Processed {processed_count} pending events")
                return {
                    "success": True,
                    "processed_count": processed_count
                }

            except Exception as e:
                db_session.rollback()
                logger.error(f"Error processing pending events: {e}")
                raise

    except Exception as e:
        logger.error(f"Error in process_pending_events task: {e}")
        self.retry(countdown=600, max_retries=2)  # Retry after 10 minutes, max 2 times