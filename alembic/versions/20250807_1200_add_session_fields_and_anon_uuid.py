"""Add session fields and anon uuid

Revision ID: 20250807_1200
Revises: 20250802
Create Date: 2025-08-07 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '20250807_1200'
down_revision: Union[str, None] = '20250802'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Users: add anon_uuid
    op.add_column('users', sa.Column('anon_uuid', sa.String(), nullable=True))
    op.create_index(op.f('ix_users_anon_uuid'), 'users', ['anon_uuid'], unique=True)

    # Sessions: add personality and last_message_at
    op.add_column('sessions', sa.Column('personality', sa.String(), nullable=True))
    op.add_column('sessions', sa.Column('last_message_at', sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    # Sessions: drop columns
    op.drop_column('sessions', 'last_message_at')
    op.drop_column('sessions', 'personality')

    # Users: drop anon_uuid
    op.drop_index(op.f('ix_users_anon_uuid'), table_name='users')
    op.drop_column('users', 'anon_uuid')


