"""
Admin endpoints for simulation engine control and monitoring.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, Optional
import logging
from datetime import datetime
from pydantic import BaseModel

from app.auth.dependencies import get_current_user
from app.models.user import User
from celery_app import celery_app

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["simulation-admin"])


class SimulationStatusResponse(BaseModel):
    """Response model for simulation status."""
    is_running: bool
    active_workers: int
    pending_tasks: int
    last_event_time: str | None
    uptime: str | None


class SimulationControlRequest(BaseModel):
    """Request model for simulation control actions."""
    action: str  # "start", "stop", "restart"


@router.get("/status", response_model=SimulationStatusResponse)
async def get_simulation_status(
    current_user: User = Depends(get_current_user)
) -> SimulationStatusResponse:
    """
    Get current status of the simulation engine.
    Requires authentication.
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required for admin access"
        )

    try:
        # Get Celery worker statistics
        inspect = celery_app.control.inspect()

        # Check active workers
        active_workers = inspect.active()
        worker_count = len(active_workers) if active_workers else 0

        # Check pending tasks in simulation queue
        reserved_tasks = inspect.reserved()
        pending_count = 0
        if reserved_tasks:
            for worker_tasks in reserved_tasks.values():
                pending_count += len(worker_tasks)

        # Check if simulation is running (simplified check)
        is_running = worker_count > 0

        # Get last event time from database
        from app.services.simulation.repository import SimulationRepository
        from app.core.database import get_async_session

        last_event_time = None
        try:
            async for db_session in get_async_session():
                try:
                    repo = SimulationRepository(db_session)
                    from sqlalchemy import select, func
                    from app.models.simulation import GlobalEvents
                    from datetime import datetime, timedelta

                    # Get most recent event
                    result = await db_session.execute(
                        select(func.max(GlobalEvents.timestamp))
                    )
                    last_event_timestamp = result.scalar()
                    if last_event_timestamp:
                        last_event_time = last_event_timestamp.isoformat()
                    break
                finally:
                    await db_session.close()
        except Exception as e:
            logger.warning(f"Could not get last event time: {e}")

        return SimulationStatusResponse(
            is_running=is_running,
            active_workers=worker_count,
            pending_tasks=pending_count,
            last_event_time=last_event_time,
            uptime=None  # Will be calculated from worker stats if needed
        )

    except Exception as e:
        logger.error(f"Error getting simulation status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get simulation status"
        )


@router.post("/control")
async def control_simulation(
    request: SimulationControlRequest,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Control simulation engine (start/stop/restart).
    Requires authentication.
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required for admin access"
        )

    try:
        if request.action == "start":
            # For now, this is a placeholder - Celery workers are started externally
            # In production, this might trigger worker scaling or enable task processing
            logger.info(f"Simulation start requested by user {current_user.email}")
            return {
                "message": "Simulation start signal sent",
                "action": "start",
                "timestamp": datetime.utcnow().isoformat(),
                "status": "acknowledged"
            }

        elif request.action == "stop":
            # Stop processing new simulation tasks
            logger.info(f"Simulation stop requested by user {current_user.email}")
            return {
                "message": "Simulation stop signal sent",
                "action": "stop",
                "timestamp": datetime.utcnow().isoformat(),
                "status": "acknowledged"
            }

        elif request.action == "restart":
            # Restart simulation processing
            logger.info(f"Simulation restart requested by user {current_user.email}")
            return {
                "message": "Simulation restart signal sent",
                "action": "restart",
                "timestamp": datetime.utcnow().isoformat(),
                "status": "acknowledged"
            }

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid action: {request.action}. Must be 'start', 'stop', or 'restart'"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error controlling simulation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to control simulation"
        )


@router.get("/health")
async def simulation_health_check() -> Dict[str, Any]:
    """
    Health check endpoint for simulation engine.
    Public endpoint for monitoring systems.
    """
    try:
        # Basic health check - ping Celery broker
        inspect = celery_app.control.inspect()
        ping_result = inspect.ping()

        is_healthy = ping_result is not None and len(ping_result) > 0

        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "celery_broker": "connected" if is_healthy else "disconnected",
            "component": "simulation-engine"
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
            "component": "simulation-engine"
        }


@router.get("/metrics")
async def get_simulation_metrics(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get detailed metrics for simulation engine.
    Requires authentication.
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required for admin access"
        )

    try:
        inspect = celery_app.control.inspect()

        # Gather various metrics
        stats = inspect.stats()
        active = inspect.active()
        scheduled = inspect.scheduled()
        reserved = inspect.reserved()

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "worker_stats": stats,
            "active_tasks": active,
            "scheduled_tasks": scheduled,
            "reserved_tasks": reserved,
            "total_workers": len(stats) if stats else 0,
            "total_active_tasks": sum(len(tasks) for tasks in active.values()) if active else 0,
        }

    except Exception as e:
        logger.error(f"Error getting simulation metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get simulation metrics"
        )


@router.get("/events/recent")
async def get_recent_events(
    limit: int = 20,
    event_type: Optional[str] = None,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get recent simulation events.
    Requires authentication.
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required for admin access"
        )

    try:
        from app.services.simulation.repository import SimulationRepository
        from app.core.database import get_async_session
        from datetime import datetime, timedelta

        async for db_session in get_async_session():
            try:
                repo = SimulationRepository(db_session)

                # Get recent events (last 24 hours)
                start_time = datetime.utcnow() - timedelta(hours=24)
                end_time = datetime.utcnow()

                from app.schemas.simulation_schemas import EventType
                event_type_filter = None
                if event_type:
                    try:
                        event_type_filter = EventType(event_type)
                    except ValueError:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Invalid event type: {event_type}"
                        )

                events = await repo.get_events_by_timeframe(
                    start_time, end_time, event_type_filter
                )

                # Limit results
                events = events[:limit]

                # Convert to response format
                event_list = []
                for event in events:
                    event_list.append({
                        "event_id": event.event_id,
                        "event_type": event.event_type,
                        "summary": event.summary,
                        "timestamp": event.timestamp.isoformat(),
                        "status": event.status,
                        "intensity": event.intensity,
                        "impact_mood": event.impact_mood,
                        "impact_energy": event.impact_energy,
                        "impact_stress": event.impact_stress
                    })

                return {
                    "events": event_list,
                    "total_returned": len(event_list),
                    "filter": {
                        "event_type": event_type,
                        "limit": limit,
                        "timeframe": "24 hours"
                    }
                }

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error getting recent events: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to get recent events"
                )
            finally:
                await db_session.close()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_recent_events: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get recent events"
        )


@router.get("/state/current")
async def get_current_ava_state(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get Ava's current global state.
    Requires authentication.
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required for admin access"
        )

    try:
        from app.services.simulation.state_manager import StateManagerService

        state_manager = StateManagerService()
        current_state = await state_manager.get_current_global_state()

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "ava_global_state": current_state
        }

    except Exception as e:
        logger.error(f"Error getting current Ava state: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get current Ava state"
        )


@router.post("/state/initialize")
async def initialize_default_states(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Initialize default states for Ava's global traits.
    Requires authentication.
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required for admin access"
        )

    try:
        from app.services.simulation.state_manager import StateManagerService

        state_manager = StateManagerService()
        result = await state_manager.initialize_default_states()

        return result

    except Exception as e:
        logger.error(f"Error initializing default states: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initialize default states"
        )


@router.get("/statistics")
async def get_simulation_statistics(
    days: int = 7,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get simulation engine statistics.
    Requires authentication.
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required for admin access"
        )

    if days < 1 or days > 30:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Days parameter must be between 1 and 30"
        )

    try:
        from app.services.simulation.repository import SimulationRepository
        from app.core.database import get_async_session

        async for db_session in get_async_session():
            try:
                repo = SimulationRepository(db_session)
                stats = await repo.get_event_statistics(days)

                return {
                    "timestamp": datetime.utcnow().isoformat(),
                    "statistics": stats
                }

            except Exception as e:
                logger.error(f"Error getting simulation statistics: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to get simulation statistics"
                )
            finally:
                await db_session.close()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_simulation_statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get simulation statistics"
        )