"""
One-off: embed every conversation_log row that has no embedding yet.

    docker compose exec -T backend python scripts/backfill_embeddings.py

Re-runnable — it only ever looks at rows WHERE embedding IS NULL, so a crash
halfway costs you the current batch and nothing else.
"""

import asyncio
import logging
import sys
import time

from sqlalchemy import text

sys.path.insert(0, "/app")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("backfill")
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

BATCH = 40
# The free-tier "100 requests/min" quota counts each *text* in a batch, not each
# HTTP call — batching saves round-trips, not quota. So pace on texts and leave
# headroom for live conversation turns embedding at the same time.
PAUSE_S = BATCH * 60 / 80


async def main() -> int:
    from app.core.database import AsyncSessionLocal
    from app.services.embeddings import embed_texts, to_pgvector

    embedded = failed = 0
    started = time.time()

    while True:
        async with AsyncSessionLocal() as session:
            rows = (await session.execute(text(
                "SELECT id, content FROM conversation_log "
                "WHERE embedding IS NULL ORDER BY id LIMIT :n"
            ), {"n": BATCH})).all()

            if not rows:
                break

            vectors = await embed_texts([r.content for r in rows])
            if vectors is None or len(vectors) != len(rows):
                # Almost always the per-minute quota. One patient retry, then give
                # up — the rows stay NULL and the next run picks them up.
                logger.warning("Batch of %d failed, waiting 60s for quota", len(rows))
                await asyncio.sleep(60)
                vectors = await embed_texts([r.content for r in rows])
            if vectors is None or len(vectors) != len(rows):
                failed += len(rows)
                logger.error("Batch of %d failed twice, stopping", len(rows))
                break

            await session.execute(
                text("UPDATE conversation_log SET embedding = CAST(:v AS vector) WHERE id = :id"),
                [{"v": to_pgvector(vec), "id": row.id} for row, vec in zip(rows, vectors)],
            )
            await session.commit()
            embedded += len(rows)
            logger.info("Embedded %d rows (%d total)", len(rows), embedded)

        await asyncio.sleep(PAUSE_S)

    async with AsyncSessionLocal() as session:
        remaining = (await session.execute(
            text("SELECT count(*) FROM conversation_log WHERE embedding IS NULL")
        )).scalar_one()

    logger.info(
        "Done in %.1fs: %d embedded, %d failed, %d still NULL",
        time.time() - started, embedded, failed, remaining,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
