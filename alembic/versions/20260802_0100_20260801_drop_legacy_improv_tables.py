"""drop_legacy_improv_tables

Drops all tables belonging to the legacy Improv Today vocabulary-practice
product (and the never-used session-state tables). Clara-only from here.

Revision ID: 20260801
Revises: 20250919
Create Date: 2026-08-02 01:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '20260801'
down_revision: Union[str, None] = '20250919'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Children before parents; CASCADE cleans up any out-of-band FKs.
LEGACY_TABLES = [
    "vocabulary_usage",
    "vocabulary_recommendations",
    "vocabulary_assessments",
    "vocabulary_suggestions",
    "session_transcripts",
    "messages",
    "conversations",
    "sessions",
    "session_states",
    "user_preferences",
    "user_session_states",
    "state_change_history",
    "session_state_backups",
]

LEGACY_USER_COLUMNS = [
    "vocabulary_tier",
    "assessment_completed",
    "interests",
]


def upgrade() -> None:
    for table in LEGACY_TABLES:
        op.execute(f'DROP TABLE IF EXISTS {table} CASCADE')

    for column in LEGACY_USER_COLUMNS:
        op.execute(f'ALTER TABLE users DROP COLUMN IF EXISTS {column}')


def downgrade() -> None:
    raise NotImplementedError("Legacy improv tables are gone for good")
