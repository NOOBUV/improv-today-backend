"""Add vocabulary_suggestions table for production

Revision ID: 57d6fcee68c8
Revises: 20250808
Create Date: 2025-08-17 01:20:13.747903

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID


# revision identifiers, used by Alembic.
revision: str = '57d6fcee68c8'
down_revision: Union[str, None] = '20250808'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create vocabulary_suggestions table only if it doesn't exist
    # (Safe for both fresh production and development environments)
    op.execute("""
        CREATE TABLE IF NOT EXISTS vocabulary_suggestions (
            id SERIAL PRIMARY KEY,
            conversation_id UUID NOT NULL REFERENCES conversations(id),
            user_id VARCHAR(255) NOT NULL,
            suggested_word VARCHAR NOT NULL,
            status VARCHAR DEFAULT 'shown',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            updated_at TIMESTAMP WITH TIME ZONE
        )
    """)
    
    # Create index if it doesn't exist
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_vocabulary_suggestions_id 
        ON vocabulary_suggestions (id)
    """)


def downgrade() -> None:
    # Drop the table
    op.drop_table('vocabulary_suggestions')
