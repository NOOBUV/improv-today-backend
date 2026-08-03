"""add pgvector embedding column to conversation_log

Semantic recall: get_related_past_snippets embeds the incoming message and
cosine-searches this column, falling back to the keyword ILIKE query whenever
the column is empty or the embedding call fails.

Nullable on purpose — a row with no embedding is a row the keyword path still
finds, and backfill (scripts/backfill_embeddings.py) fills them in later.

Revision ID: 20260803
Revises: 20260802
Create Date: 2026-08-03 03:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '20260803'
down_revision: Union[str, None] = '20260802'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("ALTER TABLE conversation_log ADD COLUMN embedding vector(768)")
    # ponytail: no ANN index. A few hundred rows scan in ~1ms and both ivfflat and
    # hnsw cost more to maintain per insert than they save here. Add
    #   CREATE INDEX ON conversation_log USING hnsw (embedding vector_cosine_ops)
    # when the table passes ~50k rows or recall latency shows up in traces.


def downgrade() -> None:
    op.execute("ALTER TABLE conversation_log DROP COLUMN embedding")
