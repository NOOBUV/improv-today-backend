"""add conversation_log table

Durable, append-only transcript of conversation turns (Redis session state
expires after 24h; this does not).

Revision ID: 20260802
Revises: 20260801
Create Date: 2026-08-02 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = '20260802'
down_revision: Union[str, None] = '20260801'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'conversation_log',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('conversation_id', sa.String(), nullable=False),
        sa.Column('role', sa.String(20), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('meta', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
    )
    op.create_index('ix_conversation_log_id', 'conversation_log', ['id'])
    op.create_index('ix_conversation_log_user_id', 'conversation_log', ['user_id'])
    op.create_index('ix_conversation_log_conversation_id', 'conversation_log', ['conversation_id'])
    op.create_index('ix_conversation_log_created_at', 'conversation_log', ['created_at'])


def downgrade() -> None:
    op.drop_table('conversation_log')
