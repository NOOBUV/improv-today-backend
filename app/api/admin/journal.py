"""
Admin API endpoints for journal entry management.
Provides secure access for reviewing and managing journal entries.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import date, datetime, timedelta, timezone
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, and_, func, update, delete
from sqlalchemy.orm import selectinload

from app.core.database import get_async_session
from app.core.auth import get_current_admin_user
from app.models.journal import JournalEntries, JournalGenerationLog, JournalTemplate
from app.schemas.journal_schemas import (
    JournalEntryResponse,
    JournalEntryListResponse,
    JournalEntryUpdate,
    JournalGenerationRequest,
    JournalGenerationResponse,
    JournalStatsResponse,
    DailyContextResponse,
    GenerationLogResponse,
    JournalTemplateResponse,
    JournalTemplateCreate,
    JournalTemplateUpdate,
    JournalStatus
)
from app.services.journal.journal_generator_service import JournalGeneratorService
from app.services.journal.daily_aggregator import DailyAggregatorService
from celery_app import celery_app

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/journal", tags=["admin", "journal"])
security = HTTPBearer()


@router.get("/entries", response_model=JournalEntryListResponse)
async def list_journal_entries(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(20, ge=1, le=100, description="Page size"),
    status: Optional[JournalStatus] = Query(None, description="Filter by status"),
    start_date: Optional[date] = Query(None, description="Start date filter"),
    end_date: Optional[date] = Query(None, description="End date filter"),
    session: AsyncSession = Depends(get_async_session),
    current_user: dict = Depends(get_current_admin_user)
):
    """List journal entries with pagination and filtering"""
    try:
        # Build query with filters
        query = select(JournalEntries)

        if status:
            query = query.where(JournalEntries.status == status.value)

        if start_date:
            query = query.where(JournalEntries.entry_date >= start_date)

        if end_date:
            query = query.where(JournalEntries.entry_date <= end_date)

        # Get total count
        count_result = await session.execute(
            select(func.count()).select_from(query.subquery())
        )
        total = count_result.scalar()

        # Apply pagination and ordering
        query = query.order_by(desc(JournalEntries.entry_date))
        query = query.offset((page - 1) * size).limit(size)

        result = await session.execute(query)
        entries = result.scalars().all()

        return JournalEntryListResponse(
            entries=[JournalEntryResponse.from_orm(entry) for entry in entries],
            total=total,
            page=page,
            size=size,
            has_next=(page * size) < total,
            has_prev=page > 1
        )

    except Exception as e:
        logger.error(f"Error listing journal entries: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve journal entries"
        )


@router.get("/entries/{entry_id}", response_model=JournalEntryResponse)
async def get_journal_entry(
    entry_id: str,
    session: AsyncSession = Depends(get_async_session),
    current_user: dict = Depends(get_current_admin_user)
):
    """Get specific journal entry by ID"""
    try:
        result = await session.execute(
            select(JournalEntries).where(JournalEntries.entry_id == entry_id)
        )
        entry = result.scalar_one_or_none()

        if not entry:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Journal entry not found"
            )

        return JournalEntryResponse.from_orm(entry)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting journal entry {entry_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve journal entry"
        )


@router.put("/entries/{entry_id}", response_model=JournalEntryResponse)
async def update_journal_entry(
    entry_id: str,
    updates: JournalEntryUpdate,
    session: AsyncSession = Depends(get_async_session),
    current_user: dict = Depends(get_current_admin_user)
):
    """Update journal entry (content, status, admin notes)"""
    try:
        # Get existing entry
        result = await session.execute(
            select(JournalEntries).where(JournalEntries.entry_id == entry_id)
        )
        entry = result.scalar_one_or_none()

        if not entry:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Journal entry not found"
            )

        # Update fields
        update_data = updates.dict(exclude_unset=True)

        if "status" in update_data:
            update_data["status"] = update_data["status"].value

        # Track reviewer
        if updates.status == JournalStatus.APPROVED:
            update_data["reviewed_at"] = datetime.now(timezone.utc)
            update_data["reviewed_by"] = current_user.get("email", "unknown")

        if updates.status == JournalStatus.POSTED:
            update_data["published_at"] = datetime.now(timezone.utc)

        # Update character count if content changed
        if "content" in update_data:
            update_data["character_count"] = len(update_data["content"])

        await session.execute(
            update(JournalEntries)
            .where(JournalEntries.entry_id == entry_id)
            .values(**update_data)
        )

        await session.commit()

        # Return updated entry
        result = await session.execute(
            select(JournalEntries).where(JournalEntries.entry_id == entry_id)
        )
        updated_entry = result.scalar_one()

        logger.info(f"Journal entry {entry_id} updated by {current_user.get('email', 'unknown')}")

        return JournalEntryResponse.from_orm(updated_entry)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating journal entry {entry_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update journal entry"
        )


@router.delete("/entries/{entry_id}")
async def delete_journal_entry(
    entry_id: str,
    session: AsyncSession = Depends(get_async_session),
    current_user: dict = Depends(get_current_admin_user)
):
    """Delete journal entry"""
    try:
        result = await session.execute(
            select(JournalEntries).where(JournalEntries.entry_id == entry_id)
        )
        entry = result.scalar_one_or_none()

        if not entry:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Journal entry not found"
            )

        await session.delete(entry)
        await session.commit()

        logger.info(f"Journal entry {entry_id} deleted by {current_user.get('email', 'unknown')}")

        return {"message": "Journal entry deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting journal entry {entry_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete journal entry"
        )


@router.post("/generate", response_model=JournalGenerationResponse)
async def generate_journal_entry(
    request: JournalGenerationRequest,
    session: AsyncSession = Depends(get_async_session),
    current_user: dict = Depends(get_current_admin_user)
):
    """Manually trigger journal generation for a specific date"""
    try:
        target_date = request.target_date or date.today()
        admin_user = current_user.get("email", "unknown")

        # Check if entry already exists
        if not request.force_regenerate:
            existing_result = await session.execute(
                select(JournalEntries).where(JournalEntries.entry_date == target_date)
            )
            existing = existing_result.scalar_one_or_none()

            if existing:
                return JournalGenerationResponse(
                    success=True,
                    entry_id=existing.entry_id,
                    target_date=target_date,
                    events_processed=existing.events_processed or 0,
                    created_new=False
                )

        # Trigger manual generation via Celery
        task = celery_app.send_task(
            "journal.manual_generate",
            args=[target_date.isoformat(), admin_user, request.force_regenerate]
        )

        # Wait for task completion (with timeout)
        result = task.get(timeout=60)

        if result["success"]:
            logger.info(f"Manual journal generation completed for {target_date}")
            return JournalGenerationResponse(
                success=True,
                entry_id=result.get("entry_id"),
                target_date=target_date,
                events_processed=result.get("events_processed", 0),
                generation_duration_ms=result.get("duration_ms"),
                created_new=True
            )
        else:
            return JournalGenerationResponse(
                success=False,
                target_date=target_date,
                events_processed=0,
                error_message=result.get("error", "Generation failed")
            )

    except Exception as e:
        logger.error(f"Error triggering journal generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate journal entry: {str(e)}"
        )


@router.get("/stats", response_model=JournalStatsResponse)
async def get_journal_stats(
    session: AsyncSession = Depends(get_async_session),
    current_user: dict = Depends(get_current_admin_user)
):
    """Get journal generation statistics"""
    try:
        # Count entries by status
        status_counts = {}
        for status_value in ["draft", "approved", "posted"]:
            result = await session.execute(
                select(func.count())
                .select_from(JournalEntries)
                .where(JournalEntries.status == status_value)
            )
            status_counts[status_value] = result.scalar()

        # Total entries
        total_result = await session.execute(
            select(func.count()).select_from(JournalEntries)
        )
        total_entries = total_result.scalar()

        # Average events per entry
        avg_result = await session.execute(
            select(func.avg(JournalEntries.events_processed))
            .where(JournalEntries.events_processed.is_not(None))
        )
        avg_events = avg_result.scalar() or 0.0

        # Most common emotional theme
        theme_result = await session.execute(
            select(
                JournalEntries.emotional_theme,
                func.count(JournalEntries.emotional_theme).label("count")
            )
            .where(JournalEntries.emotional_theme.is_not(None))
            .group_by(JournalEntries.emotional_theme)
            .order_by(desc("count"))
            .limit(1)
        )
        most_common_theme = theme_result.first()

        # Recent generation success rate (last 7 days)
        week_ago = datetime.now(timezone.utc) - timedelta(days=7)
        recent_logs_result = await session.execute(
            select(JournalGenerationLog.status)
            .where(JournalGenerationLog.created_at >= week_ago)
        )
        recent_logs = recent_logs_result.scalars().all()

        success_rate = 0.0
        if recent_logs:
            success_count = len([log for log in recent_logs if log == "success"])
            success_rate = success_count / len(recent_logs)

        # Last generated entry
        last_entry_result = await session.execute(
            select(JournalEntries.generated_at)
            .order_by(desc(JournalEntries.generated_at))
            .limit(1)
        )
        last_generated = last_entry_result.scalar_one_or_none()

        return JournalStatsResponse(
            total_entries=total_entries,
            draft_count=status_counts["draft"],
            approved_count=status_counts["approved"],
            posted_count=status_counts["posted"],
            avg_events_per_entry=float(avg_events),
            most_common_theme=most_common_theme[0] if most_common_theme else None,
            recent_generation_success_rate=float(success_rate),
            last_generated=last_generated
        )

    except Exception as e:
        logger.error(f"Error getting journal stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve statistics"
        )


@router.get("/context/{target_date}", response_model=DailyContextResponse)
async def get_daily_context(
    target_date: date,
    session: AsyncSession = Depends(get_async_session),
    current_user: dict = Depends(get_current_admin_user)
):
    """Get daily context aggregation for preview/debugging"""
    try:
        aggregator = DailyAggregatorService()
        context = await aggregator.aggregate_daily_events(session, target_date)

        return DailyContextResponse(
            target_date=target_date,
            event_count=context["total_events"],
            significant_events=context["significant_events"],
            emotional_state=context["emotional_context"],
            dominant_emotion=context["dominant_emotion"],
            emotional_arc=context["emotional_arc"]
        )

    except Exception as e:
        logger.error(f"Error getting daily context for {target_date}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve daily context"
        )


@router.get("/logs", response_model=List[GenerationLogResponse])
async def get_generation_logs(
    limit: int = Query(50, ge=1, le=200, description="Number of logs to return"),
    status_filter: Optional[str] = Query(None, description="Filter by status"),
    session: AsyncSession = Depends(get_async_session),
    current_user: dict = Depends(get_current_admin_user)
):
    """Get journal generation logs for monitoring"""
    try:
        query = select(JournalGenerationLog).order_by(desc(JournalGenerationLog.created_at))

        if status_filter:
            query = query.where(JournalGenerationLog.status == status_filter)

        query = query.limit(limit)

        result = await session.execute(query)
        logs = result.scalars().all()

        return [GenerationLogResponse.from_orm(log) for log in logs]

    except Exception as e:
        logger.error(f"Error getting generation logs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve generation logs"
        )