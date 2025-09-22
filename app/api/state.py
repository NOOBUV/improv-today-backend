"""
State management API endpoints for global and session state queries.
Provides REST endpoints for retrieving current state, state history, and session management.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Depends, Query, Path
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field

from app.core.auth import verify_token, get_current_user
from app.services.simulation.state_manager import StateManagerService
from app.services.session_state_service import SessionStateService
from app.services.state_influence_service import StateInfluenceService, ConversationScenario

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/state", tags=["state"])
security = HTTPBearer()


# Pydantic models for request/response
class StateResponse(BaseModel):
    """Response model for state data."""
    success: bool
    data: Dict[str, Any]
    timestamp: str
    message: Optional[str] = None


class SessionStateCreateRequest(BaseModel):
    """Request model for creating session state."""
    user_id: str = Field(..., description="Unique identifier for the user")
    conversation_id: str = Field(..., description="Conversation session identifier")
    personalization_data: Optional[Dict[str, Any]] = Field(None, description="User-specific preferences")


class SessionAdjustmentRequest(BaseModel):
    """Request model for updating session adjustments."""
    trait_adjustments: Dict[str, Any] = Field(..., description="Trait adjustments to apply")


class ConversationContextRequest(BaseModel):
    """Request model for building conversation context."""
    scenario: str = Field(default="casual_chat", description="Conversation scenario type")
    user_preferences: Optional[Dict[str, Any]] = Field(None, description="User preferences for state influence")


class StateHistoryResponse(BaseModel):
    """Response model for state history data."""
    success: bool
    data: List[Dict[str, Any]]
    trait_name: Optional[str]
    hours_back: int
    total_entries: int


# Global State Endpoints

@router.get("/global/current", response_model=StateResponse)
async def get_current_global_state(
    token: str = Depends(security)
) -> StateResponse:
    """
    Get Ava's current global state summary.

    Returns current values for all core traits with trends and last update info.
    """
    try:
        # Verify authentication
        await verify_token(token.credentials)

        state_manager = StateManagerService()
        global_state = await state_manager.get_current_global_state()

        logger.info("Retrieved current global state")
        return StateResponse(
            success=True,
            data=global_state,
            timestamp=datetime.now(timezone.utc).isoformat(),
            message="Current global state retrieved successfully"
        )

    except Exception as e:
        logger.error(f"Error getting current global state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/global/history", response_model=StateHistoryResponse)
async def get_state_history(
    trait_name: Optional[str] = Query(None, description="Specific trait to get history for"),
    hours_back: int = Query(24, ge=1, le=168, description="Hours of history to retrieve (1-168)"),
    token: str = Depends(security)
) -> StateHistoryResponse:
    """
    Get time-series state tracking data for analysis.

    Args:
        trait_name: Specific trait to get history for (None for all traits)
        hours_back: Number of hours of history to retrieve (1-168 hours)
    """
    try:
        # Verify authentication
        await verify_token(token.credentials)

        state_manager = StateManagerService()
        history_data = await state_manager.get_state_history(trait_name, hours_back)

        logger.info(f"Retrieved state history: {trait_name or 'all traits'}, {hours_back} hours")
        return StateHistoryResponse(
            success=True,
            data=history_data,
            trait_name=trait_name,
            hours_back=hours_back,
            total_entries=len(history_data)
        )

    except Exception as e:
        logger.error(f"Error getting state history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Session State Endpoints

@router.post("/session/create", response_model=StateResponse)
async def create_session_state(
    request: SessionStateCreateRequest,
    token: str = Depends(security)
) -> StateResponse:
    """
    Create new session state initialized from global state.

    Creates a new session state for the specified user and conversation,
    initialized with current global state as baseline.
    """
    try:
        # Verify authentication
        await verify_token(token.credentials)

        session_service = SessionStateService()
        result = await session_service.create_session_state(
            request.user_id,
            request.conversation_id,
            request.personalization_data
        )

        if result["success"]:
            logger.info(f"Created session state for user {request.user_id}")
            return StateResponse(
                success=True,
                data=result,
                timestamp=datetime.now(timezone.utc).isoformat(),
                message="Session state created successfully"
            )
        else:
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to create session state"))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating session state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{user_id}/{conversation_id}", response_model=StateResponse)
async def get_session_state(
    user_id: str = Path(..., description="User identifier"),
    conversation_id: str = Path(..., description="Conversation identifier"),
    token: str = Depends(security)
) -> StateResponse:
    """
    Retrieve session state for a specific user conversation.

    Returns the complete session state including global baseline,
    session adjustments, and conversation context.
    """
    try:
        # Verify authentication
        await verify_token(token.credentials)

        session_service = SessionStateService()
        session_state = await session_service.get_session_state(user_id, conversation_id)

        if session_state:
            logger.info(f"Retrieved session state for user {user_id}")
            return StateResponse(
                success=True,
                data=session_state,
                timestamp=datetime.now(timezone.utc).isoformat(),
                message="Session state retrieved successfully"
            )
        else:
            raise HTTPException(status_code=404, detail="Session state not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{user_id}/{conversation_id}/effective", response_model=StateResponse)
async def get_effective_state(
    user_id: str = Path(..., description="User identifier"),
    conversation_id: str = Path(..., description="Conversation identifier"),
    token: str = Depends(security)
) -> StateResponse:
    """
    Get effective state by merging global state with session adjustments.

    Returns the actual state values that should be used for conversation context,
    combining global baseline with session-specific adjustments.
    """
    try:
        # Verify authentication
        await verify_token(token.credentials)

        session_service = SessionStateService()
        effective_state = await session_service.get_effective_state(user_id, conversation_id)

        logger.info(f"Retrieved effective state for user {user_id}")
        return StateResponse(
            success=True,
            data=effective_state,
            timestamp=datetime.now(timezone.utc).isoformat(),
            message="Effective state retrieved successfully"
        )

    except Exception as e:
        logger.error(f"Error getting effective state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/session/{user_id}/{conversation_id}/adjustments", response_model=StateResponse)
async def update_session_adjustments(
    request: SessionAdjustmentRequest,
    user_id: str = Path(..., description="User identifier"),
    conversation_id: str = Path(..., description="Conversation identifier"),
    token: str = Depends(security)
) -> StateResponse:
    """
    Update session-specific state adjustments based on conversation.

    Allows for real-time adjustment of state values for this specific
    conversation session without affecting global state.
    """
    try:
        # Verify authentication
        await verify_token(token.credentials)

        session_service = SessionStateService()
        success = await session_service.update_session_adjustments(
            user_id,
            conversation_id,
            request.trait_adjustments
        )

        if success:
            logger.info(f"Updated session adjustments for user {user_id}")
            return StateResponse(
                success=True,
                data={"updated_traits": list(request.trait_adjustments.keys())},
                timestamp=datetime.now(timezone.utc).isoformat(),
                message="Session adjustments updated successfully"
            )
        else:
            raise HTTPException(status_code=400, detail="Failed to update session adjustments")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating session adjustments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{user_id}/active", response_model=StateResponse)
async def list_active_sessions(
    user_id: str = Path(..., description="User identifier"),
    token: str = Depends(security)
) -> StateResponse:
    """
    List all active sessions for a user.

    Returns summary information for all currently active conversation
    sessions for the specified user.
    """
    try:
        # Verify authentication
        await verify_token(token.credentials)

        session_service = SessionStateService()
        active_sessions = await session_service.list_active_sessions(user_id)

        logger.info(f"Listed {len(active_sessions)} active sessions for user {user_id}")
        return StateResponse(
            success=True,
            data={"active_sessions": active_sessions, "total_count": len(active_sessions)},
            timestamp=datetime.now(timezone.utc).isoformat(),
            message=f"Found {len(active_sessions)} active sessions"
        )

    except Exception as e:
        logger.error(f"Error listing active sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/session/{user_id}/{conversation_id}", response_model=StateResponse)
async def expire_session(
    user_id: str = Path(..., description="User identifier"),
    conversation_id: str = Path(..., description="Conversation identifier"),
    token: str = Depends(security)
) -> StateResponse:
    """
    Manually expire a session and clean up its state.

    Removes the session state from storage and cleans up any associated data.
    """
    try:
        # Verify authentication
        await verify_token(token.credentials)

        session_service = SessionStateService()
        success = await session_service.expire_session(user_id, conversation_id)

        if success:
            logger.info(f"Expired session for user {user_id}, conversation {conversation_id}")
            return StateResponse(
                success=True,
                data={"expired_session": f"{user_id}:{conversation_id}"},
                timestamp=datetime.now(timezone.utc).isoformat(),
                message="Session expired successfully"
            )
        else:
            raise HTTPException(status_code=400, detail="Failed to expire session")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error expiring session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# State Influence Endpoints

@router.post("/influence/context/{user_id}/{conversation_id}", response_model=StateResponse)
async def build_conversation_context(
    request: ConversationContextRequest,
    user_id: str = Path(..., description="User identifier"),
    conversation_id: str = Path(..., description="Conversation identifier"),
    token: str = Depends(security)
) -> StateResponse:
    """
    Build comprehensive conversation context by merging global and session state.

    Applies state influence algorithms to determine how current state should
    affect conversation tone, mood, and interaction style.
    """
    try:
        # Verify authentication
        await verify_token(token.credentials)

        # Validate scenario
        try:
            scenario = ConversationScenario(request.scenario)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid scenario. Must be one of: {[s.value for s in ConversationScenario]}"
            )

        influence_service = StateInfluenceService()
        context = await influence_service.build_conversation_context(
            user_id,
            conversation_id,
            scenario,
            request.user_preferences
        )

        logger.info(f"Built conversation context for user {user_id}, scenario: {request.scenario}")
        return StateResponse(
            success=True,
            data=context,
            timestamp=datetime.now(timezone.utc).isoformat(),
            message="Conversation context built successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error building conversation context: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/influence/summary/{user_id}/{conversation_id}", response_model=StateResponse)
async def get_state_influence_summary(
    user_id: str = Path(..., description="User identifier"),
    conversation_id: str = Path(..., description="Conversation identifier"),
    token: str = Depends(security)
) -> StateResponse:
    """
    Get summary of how state is influencing the current conversation.

    Returns a concise summary of primary state influences and their impact
    on conversation dynamics.
    """
    try:
        # Verify authentication
        await verify_token(token.credentials)

        influence_service = StateInfluenceService()
        summary = await influence_service.get_state_influence_summary(user_id, conversation_id)

        logger.info(f"Retrieved state influence summary for user {user_id}")
        return StateResponse(
            success=True,
            data=summary,
            timestamp=datetime.now(timezone.utc).isoformat(),
            message="State influence summary retrieved successfully"
        )

    except Exception as e:
        logger.error(f"Error getting state influence summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Admin Endpoints

@router.get("/admin/health", response_model=StateResponse)
async def state_system_health_check(
    token: str = Depends(security)
) -> StateResponse:
    """
    Check health of state management system components.

    Returns status of Redis connection, database connectivity,
    and service availability.
    """
    try:
        # Verify admin authentication (you may want to add admin role check here)
        await verify_token(token.credentials)

        session_service = SessionStateService()
        redis_health = session_service.redis_service.health_check()

        # Test state manager connectivity
        state_manager = StateManagerService()
        try:
            await state_manager.get_current_global_state()
            state_manager_health = {"connected": True, "error": None}
        except Exception as e:
            state_manager_health = {"connected": False, "error": str(e)}

        health_data = {
            "redis": redis_health,
            "state_manager": state_manager_health,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        logger.info("State system health check completed")
        return StateResponse(
            success=True,
            data=health_data,
            timestamp=datetime.now(timezone.utc).isoformat(),
            message="Health check completed"
        )

    except Exception as e:
        logger.error(f"Error in health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/admin/cleanup/sessions", response_model=StateResponse)
async def cleanup_expired_sessions(
    token: str = Depends(security)
) -> StateResponse:
    """
    Clean up expired sessions based on TTL.

    Manually trigger cleanup of expired session data.
    Admin endpoint for maintenance operations.
    """
    try:
        # Verify admin authentication
        await verify_token(token.credentials)

        session_service = SessionStateService()
        cleaned_count = await session_service.cleanup_expired_sessions()

        logger.info(f"Session cleanup completed: {cleaned_count} sessions removed")
        return StateResponse(
            success=True,
            data={"cleaned_sessions": cleaned_count},
            timestamp=datetime.now(timezone.utc).isoformat(),
            message=f"Cleaned up {cleaned_count} expired sessions"
        )

    except Exception as e:
        logger.error(f"Error in session cleanup: {e}")
        raise HTTPException(status_code=500, detail=str(e))