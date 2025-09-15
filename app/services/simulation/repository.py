"""
Repository pattern implementation for simulation engine data access.
Provides async database operations for simulation models.
"""

from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, update, delete, func, and_, or_
from sqlalchemy.orm import selectinload
from datetime import datetime, timedelta
import logging

from app.models.simulation import GlobalEvents, AvaGlobalState, SimulationLog, SimulationConfig
from app.schemas.simulation_schemas import (
    GlobalEventCreate, GlobalEventUpdate, AvaGlobalStateCreate, AvaGlobalStateUpdate,
    SimulationLogCreate, SimulationConfigCreate, SimulationConfigUpdate,
    EventType, EventStatus
)

logger = logging.getLogger(__name__)


class SimulationRepository:
    """Repository for simulation engine database operations."""

    def __init__(self, db_session: AsyncSession):
        self.db = db_session

    # Global Events operations
    async def create_global_event(self, event_data: GlobalEventCreate) -> GlobalEvents:
        """Create a new global event."""
        try:
            db_event = GlobalEvents(**event_data.model_dump())
            self.db.add(db_event)
            await self.db.commit()
            await self.db.refresh(db_event)
            return db_event
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error creating global event: {e}")
            raise

    async def get_global_event(self, event_id: str) -> Optional[GlobalEvents]:
        """Get a global event by ID."""
        try:
            result = await self.db.execute(
                select(GlobalEvents).where(GlobalEvents.event_id == event_id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting global event {event_id}: {e}")
            raise

    async def get_unprocessed_events(self, limit: int = 50) -> List[GlobalEvents]:
        """Get unprocessed global events."""
        try:
            result = await self.db.execute(
                select(GlobalEvents)
                .where(GlobalEvents.status == EventStatus.UNPROCESSED)
                .order_by(GlobalEvents.timestamp)
                .limit(limit)
            )
            return list(result.scalars().all())
        except Exception as e:
            logger.error(f"Error getting unprocessed events: {e}")
            raise

    async def get_events_by_timeframe(
        self,
        start_time: datetime,
        end_time: datetime,
        event_type: Optional[EventType] = None
    ) -> List[GlobalEvents]:
        """Get events within a time range, optionally filtered by type."""
        try:
            query = select(GlobalEvents).where(
                and_(
                    GlobalEvents.timestamp >= start_time,
                    GlobalEvents.timestamp <= end_time
                )
            )

            if event_type:
                query = query.where(GlobalEvents.event_type == event_type)

            query = query.order_by(GlobalEvents.timestamp.desc())

            result = await self.db.execute(query)
            return list(result.scalars().all())
        except Exception as e:
            logger.error(f"Error getting events by timeframe: {e}")
            raise

    async def update_global_event(self, event_id: str, event_update: GlobalEventUpdate) -> Optional[GlobalEvents]:
        """Update a global event."""
        try:
            update_data = event_update.model_dump(exclude_unset=True)
            if update_data:
                await self.db.execute(
                    update(GlobalEvents)
                    .where(GlobalEvents.event_id == event_id)
                    .values(**update_data)
                )
                await self.db.commit()

            return await self.get_global_event(event_id)
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error updating global event {event_id}: {e}")
            raise

    # Ava Global State operations
    async def get_ava_global_state(self, trait_name: str) -> Optional[AvaGlobalState]:
        """Get Ava's global state for a specific trait."""
        try:
            result = await self.db.execute(
                select(AvaGlobalState).where(AvaGlobalState.trait_name == trait_name)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting Ava global state for {trait_name}: {e}")
            raise

    async def get_all_ava_global_states(self) -> List[AvaGlobalState]:
        """Get all Ava global state traits."""
        try:
            result = await self.db.execute(
                select(AvaGlobalState).order_by(AvaGlobalState.trait_name)
            )
            return list(result.scalars().all())
        except Exception as e:
            logger.error(f"Error getting all Ava global states: {e}")
            raise

    async def create_or_update_ava_state(
        self,
        trait_name: str,
        state_data: AvaGlobalStateCreate | AvaGlobalStateUpdate
    ) -> AvaGlobalState:
        """Create or update Ava's global state for a trait."""
        try:
            existing_state = await self.get_ava_global_state(trait_name)

            if existing_state:
                # Update existing state
                update_data = state_data.model_dump(exclude_unset=True)
                if update_data:
                    await self.db.execute(
                        update(AvaGlobalState)
                        .where(AvaGlobalState.trait_name == trait_name)
                        .values(**update_data)
                    )
                    await self.db.commit()
                return await self.get_ava_global_state(trait_name)
            else:
                # Create new state
                if isinstance(state_data, AvaGlobalStateUpdate):
                    # Convert update to create data
                    create_data = AvaGlobalStateCreate(
                        trait_name=trait_name,
                        **state_data.model_dump(exclude_unset=True)
                    )
                else:
                    create_data = state_data

                db_state = AvaGlobalState(**create_data.model_dump())
                self.db.add(db_state)
                await self.db.commit()
                await self.db.refresh(db_state)
                return db_state

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error creating/updating Ava state for {trait_name}: {e}")
            raise

    # Simulation Log operations
    async def create_simulation_log(self, log_data: SimulationLogCreate) -> SimulationLog:
        """Create a new simulation log entry."""
        try:
            db_log = SimulationLog(**log_data.model_dump())
            self.db.add(db_log)
            await self.db.commit()
            await self.db.refresh(db_log)
            return db_log
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error creating simulation log: {e}")
            raise

    async def get_recent_logs(
        self,
        component: Optional[str] = None,
        level: Optional[str] = None,
        limit: int = 100
    ) -> List[SimulationLog]:
        """Get recent simulation logs with optional filtering."""
        try:
            query = select(SimulationLog)

            if component:
                query = query.where(SimulationLog.component == component)
            if level:
                query = query.where(SimulationLog.level == level)

            query = query.order_by(SimulationLog.timestamp.desc()).limit(limit)

            result = await self.db.execute(query)
            return list(result.scalars().all())
        except Exception as e:
            logger.error(f"Error getting recent logs: {e}")
            raise

    # Simulation Config operations
    async def get_config(self, key: str) -> Optional[SimulationConfig]:
        """Get a configuration setting by key."""
        try:
            result = await self.db.execute(
                select(SimulationConfig)
                .where(and_(
                    SimulationConfig.key == key,
                    SimulationConfig.is_active == True
                ))
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting config {key}: {e}")
            raise

    async def get_all_configs(self, category: Optional[str] = None) -> List[SimulationConfig]:
        """Get all active configuration settings, optionally filtered by category."""
        try:
            query = select(SimulationConfig).where(SimulationConfig.is_active == True)

            if category:
                query = query.where(SimulationConfig.category == category)

            query = query.order_by(SimulationConfig.category, SimulationConfig.key)

            result = await self.db.execute(query)
            return list(result.scalars().all())
        except Exception as e:
            logger.error(f"Error getting all configs: {e}")
            raise

    async def create_or_update_config(
        self,
        key: str,
        config_data: SimulationConfigCreate | SimulationConfigUpdate
    ) -> SimulationConfig:
        """Create or update a configuration setting."""
        try:
            existing_config = await self.get_config(key)

            if existing_config:
                # Update existing config
                update_data = config_data.model_dump(exclude_unset=True)
                if update_data:
                    await self.db.execute(
                        update(SimulationConfig)
                        .where(SimulationConfig.key == key)
                        .values(**update_data)
                    )
                    await self.db.commit()
                return await self.get_config(key)
            else:
                # Create new config
                if isinstance(config_data, SimulationConfigUpdate):
                    # Convert update to create data
                    create_data = SimulationConfigCreate(
                        key=key,
                        value=config_data.value or "",
                        **config_data.model_dump(exclude_unset=True, exclude={'value'})
                    )
                else:
                    create_data = config_data

                db_config = SimulationConfig(**create_data.model_dump())
                self.db.add(db_config)
                await self.db.commit()
                await self.db.refresh(db_config)
                return db_config

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error creating/updating config {key}: {e}")
            raise

    # Statistics and aggregation methods
    async def get_event_statistics(self, days: int = 7) -> Dict[str, Any]:
        """Get event generation statistics for the last N days."""
        try:
            start_date = datetime.utcnow() - timedelta(days=days)

            # Total events count
            total_result = await self.db.execute(
                select(func.count(GlobalEvents.event_id))
                .where(GlobalEvents.timestamp >= start_date)
            )
            total_events = total_result.scalar() or 0

            # Events by type
            type_result = await self.db.execute(
                select(GlobalEvents.event_type, func.count(GlobalEvents.event_id))
                .where(GlobalEvents.timestamp >= start_date)
                .group_by(GlobalEvents.event_type)
            )
            events_by_type = dict(type_result.fetchall())

            # Unprocessed events
            unprocessed_result = await self.db.execute(
                select(func.count(GlobalEvents.event_id))
                .where(and_(
                    GlobalEvents.timestamp >= start_date,
                    GlobalEvents.status == EventStatus.UNPROCESSED
                ))
            )
            unprocessed_events = unprocessed_result.scalar() or 0

            return {
                "total_events": total_events,
                "events_by_type": events_by_type,
                "unprocessed_events": unprocessed_events,
                "avg_events_per_day": round(total_events / days, 2) if days > 0 else 0,
                "period_days": days
            }

        except Exception as e:
            logger.error(f"Error getting event statistics: {e}")
            raise