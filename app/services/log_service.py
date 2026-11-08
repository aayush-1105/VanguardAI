import logging

from app.db.database import SessionLocal
from app.db.models import RequestLog

logger = logging.getLogger(__name__)

# shortcut: rough estimate from character count; replace with real usage tokens
# from the provider response when accurate billing matters.
CHARS_PER_TOKEN = 4
INPUT_PRICE_PER_MTOK = 0.50
OUTPUT_PRICE_PER_MTOK = 1.50


def estimate_cost(prompt: str, reply: str) -> float:
    input_tokens = len(prompt) / CHARS_PER_TOKEN
    output_tokens = len(reply) / CHARS_PER_TOKEN
    cost = (
        input_tokens / 1_000_000 * INPUT_PRICE_PER_MTOK
        + output_tokens / 1_000_000 * OUTPUT_PRICE_PER_MTOK
    )
    return round(cost, 8)


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
