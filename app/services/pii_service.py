import re


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
    def scrub_text(
        text: str, vault: dict[str, str] | None = None
    ) -> tuple[str, dict[str, str]]:
        """Tokenize PII in `text`, returning the text and the tag -> raw vault.

        Pass an existing vault to keep numbering and dedupe consistent across
        several messages: the same raw value always gets the same tag, and two
        different emails can never both become {{EMAIL_1}}.
        """
        vault = {} if vault is None else vault
        known = {raw: tag for tag, raw in vault.items()}

        def token_for(match: re.Match, category: str) -> str:
            raw = match.group(0)
            if raw in known:
                return known[raw]
            number = 1
            while f"{{{{{category}_{number}}}}}" in vault:
                number += 1
            tag = f"{{{{{category}_{number}}}}}"
            vault[tag] = raw
            known[raw] = tag
            return tag

        for pattern, category in PIIScrubber._PATTERNS:
            text = pattern.sub(lambda match, cat=category: token_for(match, cat), text)
        return text, vault
