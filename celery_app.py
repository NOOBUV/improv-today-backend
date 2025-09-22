"""
Celery configuration for Improv Today background tasks.
Handles the simulation engine and other async operations.
"""

from celery import Celery
from celery.schedules import crontab
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create Celery app instance
celery_app = Celery("improv_today")

# Redis configuration (using existing Redis from subscription system)
redis_url = os.getenv("REDIS_URL", "redis://localhost:6378/0")

# Celery configuration
celery_app.conf.update(
    # Broker settings
    broker_url=redis_url,
    result_backend=redis_url,

    # Task serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",

    # Timezone
    timezone="Europe/London",
    enable_utc=False,

    # Task routing
    task_routes={
        "app.services.simulation.event_generator.*": {"queue": "simulation"},
        "app.services.simulation.state_manager.*": {"queue": "simulation"},
        "journal.*": {"queue": "journal"},
    },

    # Beat schedule for periodic tasks
    beat_schedule={
        "generate-daily-event": {
            "task": "app.services.simulation.event_generator.generate_daily_event",
            "schedule": crontab(minute="0"),  # Every hour
            "options": {"queue": "simulation"},
        },
        "process-pending-events": {
            "task": "app.services.simulation.event_generator.process_pending_events",
            "schedule": crontab(minute="*/15"),  # Every 15 minutes
            "options": {"queue": "simulation"},
        },
        "generate-daily-journal": {
            "task": "journal.generate_daily_entry",
            "schedule": crontab(hour=23, minute=0),  # 11 PM London time (configured above)
            "options": {"queue": "journal"},
        },
        "cleanup-journal-logs": {
            "task": "journal.cleanup_old_logs",
            "schedule": crontab(hour=2, minute=0, day_of_week=1),  # Weekly on Monday at 2 AM
            "options": {"queue": "journal"},
        },
    },

    # Worker settings
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=1000,

    # Result backend settings
    result_expires=3600,  # 1 hour

    # Error handling
    task_reject_on_worker_lost=True,
    task_ignore_result=False,
)

# Auto-discover tasks from all installed apps
celery_app.autodiscover_tasks([
    "app.services.simulation",
    "app.services.journal",
])

# Import task modules to ensure they're registered
try:
    from app.services.simulation import event_generator
    from app.services.simulation import state_manager
    from app.services.journal import tasks
except ImportError:
    # Tasks not yet implemented, will be available after implementation
    pass

if __name__ == "__main__":
    celery_app.start()