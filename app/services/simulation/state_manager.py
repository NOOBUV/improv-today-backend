"""
State management integration for the simulation engine.
Handles Ava's global state updates based on generated events.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from celery import shared_task
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio
import json

from app.core.database import SessionLocal, get_async_session
from app.schemas.simulation_schemas import AvaGlobalStateUpdate, TrendDirection
from app.services.simulation.repository import SimulationRepository
from app.models.simulation import GlobalEvents
from app.models.ava_state import AvaState

logger = logging.getLogger(__name__)


class StateManagerService:
    """Service for managing Ava's global state based on simulation events."""

    def __init__(self):
        # Core traits that are tracked by the simulation engine
        self.core_traits = {
            "stress": {"min": 0, "max": 100, "default": 50},
            "energy": {"min": 0, "max": 100, "default": 70},
            "mood": {"min": 0, "max": 100, "default": 60},
            "social_satisfaction": {"min": 0, "max": 100, "default": 60},
            "work_satisfaction": {"min": 0, "max": 100, "default": 65},
            "personal_fulfillment": {"min": 0, "max": 100, "default": 55}
        }

    async def process_event_impact(self, event: GlobalEvents) -> Dict[str, Any]:
        """
        Process an event and update Ava's global state accordingly.
        Returns a summary of state changes made.
        """
        try:
            logger.info(f"Processing event impact for event {event.event_id}: {event.event_type}")

            async for db_session in get_async_session():
                try:
                    repo = SimulationRepository(db_session)

                    # Calculate state changes based on event
                    state_changes = self._calculate_state_changes(event)

                    # Apply changes to each affected trait
                    change_summary = {}
                    for trait_name, change_info in state_changes.items():
                        updated_state = await self._update_trait_state(
                            repo, trait_name, change_info, event.event_id
                        )
                        change_summary[trait_name] = {
                            "previous_value": change_info.get("previous_value"),
                            "new_value": updated_state.numeric_value,
                            "change_amount": change_info["change_amount"],
                            "reason": change_info["reason"]
                        }

                    # Log state changes for audit trail
                    await self._log_state_changes(repo, event.event_id, change_summary)

                    logger.info(f"Applied {len(state_changes)} state changes for event {event.event_id}")
                    return {
                        "success": True,
                        "event_id": event.event_id,
                        "changes": change_summary
                    }

                except Exception as e:
                    logger.error(f"Error processing event impact: {e}")
                    raise
                finally:
                    await db_session.close()

        except Exception as e:
            logger.error(f"Error in process_event_impact: {e}")
            raise

    async def process_consciousness_emotional_reactions(
        self,
        event: GlobalEvents,
        consciousness_response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process consciousness response emotional reactions to update global state.
        This method analyzes the emotional_reaction field from consciousness generation
        and applies appropriate state adjustments beyond basic event impacts.

        Args:
            event: The source GlobalEvent that generated the consciousness response
            consciousness_response: Dict containing emotional_reaction, chosen_action, internal_thoughts

        Returns:
            Summary of additional state changes made based on emotional reactions
        """
        try:
            logger.info(f"Processing consciousness emotional reactions for event {event.event_id}")

            emotional_reaction = consciousness_response.get("emotional_reaction", "")
            internal_thoughts = consciousness_response.get("internal_thoughts", "")

            if not emotional_reaction:
                logger.warning(f"No emotional reaction found in consciousness response for event {event.event_id}")
                return {"success": False, "reason": "No emotional reaction to process"}

            db_session_gen = get_async_session()
            db_session = await db_session_gen.__anext__()
            try:
                repo = SimulationRepository(db_session)

                # Analyze emotional reactions for additional state changes
                emotional_state_changes = self._analyze_emotional_reactions(
                    emotional_reaction, internal_thoughts, event
                )

                # Apply emotional state changes
                change_summary = {}
                for trait_name, change_info in emotional_state_changes.items():
                    updated_state = await self._update_trait_state(
                        repo, trait_name, change_info, event.event_id
                    )
                    change_summary[trait_name] = {
                        "previous_value": change_info.get("previous_value"),
                        "new_value": updated_state.numeric_value,
                        "change_amount": change_info["change_amount"],
                        "reason": change_info["reason"]
                    }

                # Log emotional processing for audit trail
                await self._log_emotional_processing(
                    repo, event.event_id, emotional_reaction, change_summary
                )

                logger.info(f"Applied {len(emotional_state_changes)} emotional state changes for event {event.event_id}")
                return {
                    "success": True,
                    "event_id": event.event_id,
                    "emotional_changes": change_summary,
                    "processed_reaction": emotional_reaction[:100] + "..." if len(emotional_reaction) > 100 else emotional_reaction
                }

            except Exception as e:
                logger.error(f"Error processing emotional reactions: {e}")
                raise
            finally:
                await db_session.close()

        except Exception as e:
            logger.error(f"Error in process_consciousness_emotional_reactions: {e}")
            return {"success": False, "error": str(e)}

    def _analyze_emotional_reactions(
        self,
        emotional_reaction: str,
        internal_thoughts: str,
        event: GlobalEvents
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze emotional reactions and internal thoughts to determine additional state changes.
        This method looks for emotional keywords and patterns to apply nuanced state adjustments.
        """
        changes = {}
        emotional_text = f"{emotional_reaction} {internal_thoughts}".lower()

        # Emotional intensity patterns for mood adjustments
        intense_positive_words = ["thrilled", "ecstatic", "overjoyed", "elated", "amazing", "fantastic", "incredible"]
        intense_negative_words = ["devastated", "heartbroken", "furious", "terrified", "overwhelmed", "crushed"]
        mild_positive_words = ["happy", "pleased", "content", "satisfied", "relieved", "grateful"]
        mild_negative_words = ["disappointed", "annoyed", "concerned", "worried", "frustrated", "tired"]

        # Stress-related emotional patterns
        stress_increase_words = ["anxious", "worried", "stressed", "overwhelmed", "pressure", "panic", "tense"]
        stress_decrease_words = ["calm", "relaxed", "peaceful", "serene", "relieved", "centered"]

        # Energy-related emotional patterns
        energy_increase_words = ["excited", "energized", "motivated", "inspired", "pumped", "invigorated"]
        energy_decrease_words = ["exhausted", "drained", "tired", "depleted", "weary", "sluggish"]

        # Analyze mood changes based on emotional intensity
        mood_change = 0
        if any(word in emotional_text for word in intense_positive_words):
            mood_change = 8
        elif any(word in emotional_text for word in mild_positive_words):
            mood_change = 4
        elif any(word in emotional_text for word in intense_negative_words):
            mood_change = -10
        elif any(word in emotional_text for word in mild_negative_words):
            mood_change = -3

        if mood_change != 0:
            changes["mood"] = {
                "change_amount": mood_change,
                "reason": f"Emotional reaction analysis: {emotional_reaction[:50]}..."
            }

        # Analyze stress changes
        stress_change = 0
        if any(word in emotional_text for word in stress_increase_words):
            stress_change = 6
        elif any(word in emotional_text for word in stress_decrease_words):
            stress_change = -5

        if stress_change != 0:
            changes["stress"] = {
                "change_amount": stress_change,
                "reason": f"Stress reaction from consciousness: {emotional_reaction[:50]}..."
            }

        # Analyze energy changes
        energy_change = 0
        if any(word in emotional_text for word in energy_increase_words):
            energy_change = 5
        elif any(word in emotional_text for word in energy_decrease_words):
            energy_change = -4

        if energy_change != 0:
            changes["energy"] = {
                "change_amount": energy_change,
                "reason": f"Energy shift from consciousness: {emotional_reaction[:50]}..."
            }

        # Event-specific emotional processing
        if event.event_type == "social":
            # Social events can affect social satisfaction based on emotional tone
            if mood_change > 0:
                changes["social_satisfaction"] = {
                    "change_amount": 3,
                    "reason": "Positive social emotional experience"
                }
            elif mood_change < -5:
                changes["social_satisfaction"] = {
                    "change_amount": -4,
                    "reason": "Negative social emotional experience"
                }

        logger.debug(f"Emotional analysis found {len(changes)} additional state changes")
        return changes

    async def _log_state_changes(
        self,
        repo: Any,
        event_id: str,
        change_summary: Dict[str, Any]
    ) -> None:
        """Log state changes for audit trail and time-series tracking."""
        try:
            # Create audit log entry for state changes
            audit_data = {
                "event_id": event_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "change_type": "event_impact",
                "changes": change_summary,
                "total_changes": len(change_summary)
            }

            # Store in audit log (this will need StateChangeHistory model)
            logger.info(f"State change audit: {event_id} -> {len(change_summary)} changes")
            # TODO: Implement actual audit log storage when StateChangeHistory model is created

        except Exception as e:
            logger.error(f"Error logging state changes: {e}")

    async def _log_emotional_processing(
        self,
        repo: Any,
        event_id: str,
        emotional_reaction: str,
        change_summary: Dict[str, Any]
    ) -> None:
        """Log emotional processing results for audit trail."""
        try:
            # Create audit log entry for emotional processing
            audit_data = {
                "event_id": event_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "change_type": "emotional_processing",
                "emotional_reaction_excerpt": emotional_reaction[:200],
                "changes": change_summary,
                "total_changes": len(change_summary)
            }

            logger.info(f"Emotional processing audit: {event_id} -> {len(change_summary)} emotional changes")
            # TODO: Implement actual audit log storage when StateChangeHistory model is created

        except Exception as e:
            logger.error(f"Error logging emotional processing: {e}")

    async def get_state_history(
        self,
        trait_name: Optional[str] = None,
        hours_back: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Get time-series state tracking data for analysis.

        Args:
            trait_name: Specific trait to get history for (None for all traits)
            hours_back: Number of hours of history to retrieve

        Returns:
            List of state changes with timestamps for analysis
        """
        try:
            db_session_gen = get_async_session()
            db_session = await db_session_gen.__anext__()
            try:
                repo = SimulationRepository(db_session)

                if trait_name:
                    # Get history for specific trait
                    history = await repo.get_trait_history(trait_name, hours_back)
                else:
                    # Get history for all traits
                    history = await repo.get_all_traits_history(hours_back)

                return history

            except Exception as e:
                logger.error(f"Error getting state history: {e}")
                return []
            finally:
                await db_session.close()

        except Exception as e:
            logger.error(f"Error in get_state_history: {e}")
            return []

    def _calculate_state_changes(self, event: GlobalEvents) -> Dict[str, Dict[str, Any]]:
        """Calculate how an event should affect Ava's state traits."""
        changes = {}

        # Base change amounts based on event intensity
        intensity_multiplier = event.intensity / 5.0 if event.intensity else 1.0

        # Process mood impact
        if event.impact_mood:
            mood_change = 0
            if event.impact_mood == "positive":
                mood_change = int(5 * intensity_multiplier)
            elif event.impact_mood == "negative":
                mood_change = int(-3 * intensity_multiplier)

            if mood_change != 0:
                changes["mood"] = {
                    "change_amount": mood_change,
                    "reason": f"Event had {event.impact_mood} mood impact"
                }

        # Process energy impact
        if event.impact_energy:
            energy_change = 0
            if event.impact_energy == "increase":
                energy_change = int(4 * intensity_multiplier)
            elif event.impact_energy == "decrease":
                energy_change = int(-4 * intensity_multiplier)

            if energy_change != 0:
                changes["energy"] = {
                    "change_amount": energy_change,
                    "reason": f"Event caused energy to {event.impact_energy}"
                }

        # Process stress impact
        if event.impact_stress:
            stress_change = 0
            if event.impact_stress == "increase":
                stress_change = int(6 * intensity_multiplier)
            elif event.impact_stress == "decrease":
                stress_change = int(-5 * intensity_multiplier)

            if stress_change != 0:
                changes["stress"] = {
                    "change_amount": stress_change,
                    "reason": f"Event caused stress to {event.impact_stress}"
                }

        # Event type specific impacts
        if event.event_type == "work":
            # Work events affect work satisfaction
            work_change = int(2 * intensity_multiplier)
            if event.impact_mood == "negative":
                work_change = int(-3 * intensity_multiplier)
            elif event.impact_stress == "increase":
                work_change = int(-2 * intensity_multiplier)

            changes["work_satisfaction"] = {
                "change_amount": work_change,
                "reason": f"Work event with intensity {event.intensity}"
            }

        elif event.event_type == "social":
            # Social events affect social satisfaction
            social_change = int(4 * intensity_multiplier)
            if event.impact_mood == "negative":
                social_change = int(-2 * intensity_multiplier)

            changes["social_satisfaction"] = {
                "change_amount": social_change,
                "reason": f"Social event with {event.impact_mood or 'neutral'} mood impact"
            }

        elif event.event_type == "personal":
            # Personal events affect personal fulfillment
            personal_change = int(3 * intensity_multiplier)
            if event.impact_mood == "positive":
                personal_change = int(5 * intensity_multiplier)
            elif event.impact_mood == "negative":
                personal_change = int(-2 * intensity_multiplier)

            changes["personal_fulfillment"] = {
                "change_amount": personal_change,
                "reason": f"Personal event improved self-care"
            }

        return changes

    async def _update_trait_state(
        self,
        repo: SimulationRepository,
        trait_name: str,
        change_info: Dict[str, Any],
        event_id: str
    ) -> Any:
        """Update a specific trait state in the database."""
        try:
            # Get current state
            current_state = await repo.get_ava_global_state(trait_name)

            if current_state:
                current_value = current_state.numeric_value or self.core_traits[trait_name]["default"]
            else:
                current_value = self.core_traits[trait_name]["default"]

            # Calculate new value
            change_amount = change_info["change_amount"]
            new_value = current_value + change_amount

            # Apply bounds
            trait_config = self.core_traits.get(trait_name, {"min": 0, "max": 100})
            new_value = max(trait_config["min"], min(trait_config["max"], new_value))

            # Determine trend
            if change_amount > 0:
                trend = TrendDirection.INCREASING
            elif change_amount < 0:
                trend = TrendDirection.DECREASING
            else:
                trend = TrendDirection.STABLE

            # Store previous value for reporting
            change_info["previous_value"] = current_value

            # Update state
            update_data = AvaGlobalStateUpdate(
                value=str(new_value),
                numeric_value=new_value,
                change_reason=change_info["reason"],
                trend=trend,
                last_event_id=event_id
            )

            updated_state = await repo.create_or_update_ava_state(trait_name, update_data)
            logger.debug(f"Updated {trait_name}: {current_value} -> {new_value} ({change_amount:+d})")

            return updated_state

        except Exception as e:
            logger.error(f"Error updating trait {trait_name}: {e}")
            raise

    async def get_current_global_state(self) -> Dict[str, Any]:
        """Get Ava's current global state summary."""
        try:
            async for db_session in get_async_session():
                try:
                    repo = SimulationRepository(db_session)
                    all_states = await repo.get_all_ava_global_states()

                    state_summary = {}
                    for state in all_states:
                        state_summary[state.trait_name] = {
                            "value": state.value,
                            "numeric_value": state.numeric_value,
                            "trend": state.trend,
                            "last_updated": state.last_updated.isoformat(),
                            "last_change_reason": state.change_reason
                        }

                    # Ensure all core traits are represented
                    for trait_name, config in self.core_traits.items():
                        if trait_name not in state_summary:
                            state_summary[trait_name] = {
                                "value": str(config["default"]),
                                "numeric_value": config["default"],
                                "trend": "stable",
                                "last_updated": datetime.now(timezone.utc).isoformat(),
                                "last_change_reason": "Default value"
                            }

                    return state_summary

                except Exception as e:
                    logger.error(f"Error getting current global state: {e}")
                    raise
                finally:
                    await db_session.close()

        except Exception as e:
            logger.error(f"Error in get_current_global_state: {e}")
            raise

    async def initialize_default_states(self) -> Dict[str, Any]:
        """Initialize default states for all core traits if they don't exist."""
        try:
            async for db_session in get_async_session():
                try:
                    repo = SimulationRepository(db_session)
                    initialized_traits = []

                    for trait_name, config in self.core_traits.items():
                        existing_state = await repo.get_ava_global_state(trait_name)

                        if not existing_state:
                            from app.schemas.simulation_schemas import AvaGlobalStateCreate
                            create_data = AvaGlobalStateCreate(
                                trait_name=trait_name,
                                value=str(config["default"]),
                                numeric_value=config["default"],
                                change_reason="Initial default value",
                                trend=TrendDirection.STABLE,
                                min_value=config["min"],
                                max_value=config["max"]
                            )

                            await repo.create_or_update_ava_state(trait_name, create_data)
                            initialized_traits.append(trait_name)
                            logger.info(f"Initialized default state for {trait_name}")

                    return {
                        "success": True,
                        "initialized_traits": initialized_traits,
                        "message": f"Initialized {len(initialized_traits)} default traits"
                    }

                except Exception as e:
                    logger.error(f"Error initializing default states: {e}")
                    raise
                finally:
                    await db_session.close()

        except Exception as e:
            logger.error(f"Error in initialize_default_states: {e}")
            raise


# Celery task for processing event impacts
@shared_task(bind=True, name="app.services.simulation.state_manager.process_event_impacts")
def process_event_impacts(self):
    """
    Celery task to process event impacts on Ava's global state.
    This task is called by the event processing workflow.
    """
    try:
        logger.info("Starting batch processing of event impacts")

        async def _async_process():
            async for db_session in get_async_session():
                try:
                    repo = SimulationRepository(db_session)
                    state_manager = StateManagerService()

                    # Get recent unprocessed events
                    unprocessed_events = await repo.get_unprocessed_events(limit=5)

                    processed_count = 0
                    for event in unprocessed_events:
                        # Process state impact
                        impact_result = await state_manager.process_event_impact(event)

                        # Mark event as processed
                        from app.schemas.simulation_schemas import GlobalEventUpdate, EventStatus
                        update_data = GlobalEventUpdate(
                            status=EventStatus.PROCESSED,
                            processed_at=datetime.now(timezone.utc)
                        )
                        await repo.update_global_event(event.event_id, update_data)

                        processed_count += 1
                        logger.info(f"Processed event impact for {event.event_id}")

                    return {
                        "success": True,
                        "processed_count": processed_count
                    }

                except Exception as e:
                    logger.error(f"Error processing event impacts: {e}")
                    raise
                finally:
                    await db_session.close()

        # Execute async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(_async_process())
            return result
        finally:
            loop.close()

    except Exception as e:
        logger.error(f"Error in process_event_impacts task: {e}")
        self.retry(countdown=300, max_retries=3)