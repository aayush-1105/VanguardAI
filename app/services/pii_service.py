import logging
import os
import re
from uuid import uuid4

import redis

logger = logging.getLogger(__name__)

# A request's tokens are only needed while its response is being assembled.
VAULT_TTL_SECONDS = 300

# from_url does not connect until the first command, so importing the app works
# without Redis running.
_client = redis.Redis.from_url(
    os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True
)


class Vault:
    """The tag -> raw mapping for one request, held in Redis with a 300s TTL.

    Redis rather than a process-local dict so any worker can rehydrate a
    response and the raw values expire on their own after the TTL.
    """

    def __init__(self, key: str | None = None) -> None:
        self.key = key or f"vanguard:vault:{uuid4().hex}"
        self._local: dict[str, str] = {}

    def load(self) -> dict[str, str]:
        try:
            return _client.hgetall(self.key)
        except redis.RedisError:
            # Redis is a shared store, not a dependency worth failing a request
            # over: a single worker can still rehydrate from its own copy.
            logger.warning("redis unavailable; using the in-process PII vault")
            return dict(self._local)

    def save(self, mapping: dict[str, str]) -> None:
        if not mapping:
            return
        self._local = dict(mapping)
        try:
            pipe = _client.pipeline()
            pipe.hset(self.key, mapping=mapping)
            pipe.expire(self.key, VAULT_TTL_SECONDS)
            pipe.execute()
        except redis.RedisError:
            logger.warning("redis unavailable; PII vault kept in process only")


class PIIScrubber:
    """Replace secrets and identifiers with reversible tokens like {{EMAIL_1}}."""

    # Order matters: the widest digit patterns run last so they cannot eat a
    # narrower match first (a 16-digit card is not a phone number).
    _PATTERNS = [
        (re.compile(r"bearer\s+[A-Za-z0-9._\-]+", re.IGNORECASE), "API_KEY"),
        (re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}"), "API_KEY"),
        (re.compile(r"\bgsk_[A-Za-z0-9_\-]{20,}"), "API_KEY"),
        (re.compile(r"[\w.+\-]+@[\w\-]+\.[\w.\-]+"), "EMAIL"),
        (re.compile(r"(?<!\d)(?:\d[ \-]?){13,15}\d(?!\d)"), "CREDIT_CARD"),
        (re.compile(r"(?<!\d)\d{4}\s?\d{4}\s?\d{4}(?!\d)"), "AADHAAR"),
        (re.compile(r"(?<![\d+])(?:\+\d{1,3}[\s.\-]?)?(?:\d[\s.\-]?){9,12}\d(?!\d)"), "PHONE"),
    ]

    @staticmethod
    def scrub_text(text: str, vault: Vault | None = None) -> tuple[str, Vault]:
        """Tokenize PII in `text`, returning the text and the Redis-backed vault.

        Pass an existing vault to keep numbering and dedupe consistent across
        several messages: the same raw value always gets the same tag, and two
        different emails can never both become {{EMAIL_1}}.
        """
        vault = Vault() if vault is None else vault
        mapping = vault.load()
        known = {raw: tag for tag, raw in mapping.items()}

        def token_for(match: re.Match, category: str) -> str:
            raw = match.group(0)
            if raw in known:
                return known[raw]
            number = 1
            while f"{{{{{category}_{number}}}}}" in mapping:
                number += 1
            tag = f"{{{{{category}_{number}}}}}"
            mapping[tag] = raw
            known[raw] = tag
            return tag

        for pattern, category in PIIScrubber._PATTERNS:
            text = pattern.sub(lambda match, cat=category: token_for(match, cat), text)
        vault.save(mapping)
        return text, vault
