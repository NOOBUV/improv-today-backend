"""
Celery tasks for journal generation and management.
Handles scheduled and manual journal generation operations.
"""

import logging
from datetime import datetime, date, timedelta, timezone
from typing import Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from celery import Task
from celery_app import celery_app
from app.core.database import AsyncSessionLocal
from app.services.journal.journal_generator_service import JournalGeneratorService
from app.services.journal.daily_aggregator import DailyAggregatorService
from app.models.journal import JournalEntries, JournalGenerationLog
from app.schemas.journal_schemas import JournalEntryCreate

logger = logging.getLogger(__name__)


class DatabaseTask(Task):
    """Base task class that provides database session handling"""

    async def get_session(self) -> AsyncSession:
        """Get async database session"""
        return AsyncSessionLocal()


@celery_app.task(bind=True, base=DatabaseTask, name="journal.generate_daily_entry")
def generate_daily_journal_entry(self, target_date_str: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate daily journal entry for the specified date.
    Scheduled to run at 11 PM London time (23:00 Europe/London).

    Args:
        target_date_str: Date string in YYYY-MM-DD format (defaults to today)

    Returns:
        Dictionary containing generation results
    """
    import asyncio
    return asyncio.run(self._generate_daily_journal_entry_async(target_date_str))

    async def _generate_daily_journal_entry_async(self, target_date_str: Optional[str] = None) -> Dict[str, Any]:
    """Async implementation of daily journal generation"""
    start_time = datetime.now(timezone.utc)
    target_date = date.fromisoformat(target_date_str) if target_date_str else date.today()

    session = None
    journal_service = JournalGeneratorService()
    generation_log = None

    try:
        session = AsyncSessionLocal()

        # Check if journal entry already exists for this date
        existing_entry = await session.execute(
            select(JournalEntries).where(JournalEntries.entry_date == target_date)
        )
        existing = existing_entry.scalar_one_or_none()

        if existing:
            logger.info(f"Journal entry already exists for {target_date}, skipping generation")
            await _log_generation_attempt(
                session, target_date, "skipped", 0, 0,
                self.request.id, "scheduled",
                error_message="Entry already exists"
            )
            return {
                "success": True,
                "skipped": True,
                "target_date": target_date.isoformat(),
                "reason": "Entry already exists"
            }

        # Generate journal entry
        journal_data = await journal_service.generate_daily_journal(session, target_date)

        if not journal_data:
            logger.warning(f"No journal content generated for {target_date}")
            await _log_generation_attempt(
                session, target_date, "no_events", 0, 0,
                self.request.id, "scheduled",
                error_message="No events found for journal generation"
            )
            return {
                "success": False,
                "target_date": target_date.isoformat(),
                "reason": "No events found"
            }

        # Create journal entry in database
        journal_entry = JournalEntries(
            entry_date=journal_data["entry_date"],
            content=journal_data["content"],
            status=journal_data["status"],
            events_processed=journal_data["events_processed"],
            emotional_theme=journal_data["emotional_theme"],
            generated_at=journal_data["generated_at"],
            character_count=len(journal_data["content"])
        )

        session.add(journal_entry)
        await session.commit()
        await session.refresh(journal_entry)

        # Calculate generation duration
        duration_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)

        # Log successful generation
        await _log_generation_attempt(
            session, target_date, "success",
            journal_data["events_processed"], journal_data["events_processed"],
            self.request.id, "scheduled",
            duration_ms=duration_ms, llm_model="gpt-4o-mini"
        )

        logger.info(f"Successfully generated journal entry {journal_entry.entry_id} for {target_date}")

        return {
            "success": True,
            "entry_id": journal_entry.entry_id,
            "target_date": target_date.isoformat(),
            "events_processed": journal_data["events_processed"],
            "content_length": len(journal_data["content"]),
            "duration_ms": duration_ms
        }

    except Exception as e:
        error_msg = f"Failed to generate journal entry for {target_date}: {str(e)}"
        logger.error(error_msg, exc_info=True)

        if session:
            try:
                duration_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
                await _log_generation_attempt(
                    session, target_date, "failure", 0, 0,
                    self.request.id, "scheduled",
                    duration_ms=duration_ms, error_message=str(e)
                )
            except Exception as log_error:
                logger.error(f"Failed to log generation failure: {log_error}")

        return {
            "success": False,
            "target_date": target_date.isoformat(),
            "error": str(e)
        }

    finally:
        if session:
            await session.close()


@celery_app.task(bind=True, base=DatabaseTask, name="journal.manual_generate")
def manual_generate_journal_entry(
    self,
    target_date_str: str,
    admin_user: str,
    force_regenerate: bool = False
) -> Dict[str, Any]:
    """
    Manually generate journal entry for specified date.
    Called via admin API.

    Args:
        target_date_str: Date string in YYYY-MM-DD format
        admin_user: Admin user requesting generation
        force_regenerate: Whether to regenerate if entry exists

    Returns:
        Dictionary containing generation results
    """
    import asyncio
    return asyncio.run(
        self._manual_generate_journal_entry_async(target_date_str, admin_user, force_regenerate)
    )

    async def _manual_generate_journal_entry_async(
        self,
    target_date_str: str,
    admin_user: str,
    force_regenerate: bool = False
) -> Dict[str, Any]:
    """Async implementation of manual journal generation"""
    start_time = datetime.now(timezone.utc)
    target_date = date.fromisoformat(target_date_str)

    session = None
    journal_service = JournalGeneratorService()

    try:
        session = AsyncSessionLocal()

        # Check if journal entry already exists
        existing_entry = await session.execute(
            select(JournalEntries).where(JournalEntries.entry_date == target_date)
        )
        existing = existing_entry.scalar_one_or_none()

        if existing and not force_regenerate:
            logger.info(f"Manual generation skipped - entry exists for {target_date}")
            return {
                "success": True,
                "skipped": True,
                "entry_id": existing.entry_id,
                "target_date": target_date.isoformat(),
                "reason": "Entry already exists (use force_regenerate=true to override)"
            }

        # Delete existing entry if force regenerating
        if existing and force_regenerate:
            await session.delete(existing)
            await session.commit()
            logger.info(f"Deleted existing entry for force regeneration: {existing.entry_id}")

        # Generate journal entry
        journal_data = await journal_service.generate_daily_journal(session, target_date)

        if not journal_data:
            logger.warning(f"Manual generation failed - no content for {target_date}")
            await _log_generation_attempt(
                session, target_date, "no_events", 0, 0,
                self.request.id, "manual",
                error_message="No events found for journal generation"
            )
            return {
                "success": False,
                "target_date": target_date.isoformat(),
                "reason": "No events found for journal generation"
            }

        # Create journal entry in database
        journal_entry = JournalEntries(
            entry_date=journal_data["entry_date"],
            content=journal_data["content"],
            status=journal_data["status"],
            events_processed=journal_data["events_processed"],
            emotional_theme=journal_data["emotional_theme"],
            generated_at=journal_data["generated_at"],
            character_count=len(journal_data["content"]),
            reviewed_by=admin_user  # Mark as manually generated
        )

        session.add(journal_entry)
        await session.commit()
        await session.refresh(journal_entry)

        # Calculate generation duration
        duration_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)

        # Log successful manual generation
        await _log_generation_attempt(
            session, target_date, "success",
            journal_data["events_processed"], journal_data["events_processed"],
            self.request.id, "manual",
            duration_ms=duration_ms, llm_model="gpt-4o-mini"
        )

        logger.info(f"Successfully manually generated journal entry {journal_entry.entry_id} for {target_date}")

        return {
            "success": True,
            "entry_id": journal_entry.entry_id,
            "target_date": target_date.isoformat(),
            "events_processed": journal_data["events_processed"],
            "content_length": len(journal_data["content"]),
            "duration_ms": duration_ms,
            "regenerated": existing is not None
        }

    except Exception as e:
        error_msg = f"Manual journal generation failed for {target_date}: {str(e)}"
        logger.error(error_msg, exc_info=True)

        if session:
            try:
                duration_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
                await _log_generation_attempt(
                    session, target_date, "failure", 0, 0,
                    self.request.id, "manual",
                    duration_ms=duration_ms, error_message=str(e)
                )
            except Exception as log_error:
                logger.error(f"Failed to log manual generation failure: {log_error}")

        return {
            "success": False,
            "target_date": target_date.isoformat(),
            "error": str(e)
        }

    finally:
        if session:
            await session.close()


@celery_app.task(bind=True, base=DatabaseTask, name="journal.cleanup_old_logs")
def cleanup_old_generation_logs(self, days_to_keep: int = 30) -> Dict[str, Any]:
    """
    Clean up old journal generation logs.
    Scheduled to run weekly.

    Args:
        days_to_keep: Number of days of logs to keep

    Returns:
        Dictionary containing cleanup results
    """
    import asyncio
    return asyncio.run(self._cleanup_old_generation_logs_async(days_to_keep))

    async def _cleanup_old_generation_logs_async(self, days_to_keep: int = 30) -> Dict[str, Any]:
    """Async implementation of log cleanup"""
    session = None

    try:
        session = AsyncSessionLocal()

        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)

        # Count logs to be deleted
        count_result = await session.execute(
            select(JournalGenerationLog)
            .where(JournalGenerationLog.created_at < cutoff_date)
        )
        logs_to_delete = count_result.scalars().all()
        delete_count = len(logs_to_delete)

        if delete_count == 0:
            return {"success": True, "deleted_count": 0, "message": "No old logs to delete"}

        # Delete old logs
        for log in logs_to_delete:
            await session.delete(log)

        await session.commit()

        logger.info(f"Cleaned up {delete_count} journal generation logs older than {days_to_keep} days")

        return {
            "success": True,
            "deleted_count": delete_count,
            "cutoff_date": cutoff_date.isoformat(),
            "days_kept": days_to_keep
        }

    except Exception as e:
        error_msg = f"Failed to cleanup old generation logs: {str(e)}"
        logger.error(error_msg, exc_info=True)

        return {
            "success": False,
            "error": str(e)
        }

    finally:
        if session:
            await session.close()


async def _log_generation_attempt(
    session: AsyncSession,
    target_date: date,
    status: str,
    events_found: int,
    events_processed: int,
    celery_task_id: str,
    triggered_by: str,
    duration_ms: Optional[int] = None,
    llm_model: Optional[str] = None,
    error_message: Optional[str] = None
) -> None:
    """Log journal generation attempt to database"""
    try:
        generation_log = JournalGenerationLog(
            target_date=target_date,
            status=status,
            events_found=events_found,
            events_processed=events_processed,
            generation_duration_ms=duration_ms,
            llm_model_used=llm_model,
            error_message=error_message,
            celery_task_id=celery_task_id,
            triggered_by=triggered_by
        )

        session.add(generation_log)
        await session.commit()

    except Exception as e:
        logger.error(f"Failed to log generation attempt: {e}")


# Register tasks in Celery app
@celery_app.task(bind=True, name="journal.test_connection")
def test_journal_connection(self):
    """Test task to verify journal generation infrastructure"""
    return {
        "success": True,
        "message": "Journal generation tasks are working",
        "task_id": self.request.id,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }