import logging

from app.db.database import SessionLocal
from app.db.models import RequestLog

logger = logging.getLogger(__name__)


async def write_log(provider_used: str, cache_hit: bool, ttft_ms: float, total_cost: float) -> None:
    try:
        async with SessionLocal() as session:
            session.add(
                RequestLog(
                    provider_used=provider_used,
                    cache_hit=cache_hit,
                    ttft_ms=ttft_ms,
                    total_cost=total_cost,
                )
            )
            await session.commit()
    except Exception:
        # Telemetry must never fail a request the user may already have paid for.
        logger.exception("failed to write request log")
