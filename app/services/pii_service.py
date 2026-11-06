import re


class PIIScrubber:
    """Replace secrets and identifiers with tags before text leaves the process."""

    # Order matters: the widest digit patterns run last so they cannot eat a
    # narrower match first (a 16-digit card is not a phone number).
    _PATTERNS = [
        (re.compile(r"bearer\s+[A-Za-z0-9._\-]+", re.IGNORECASE), "[REDACTED_API_KEY]"),
        (re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}"), "[REDACTED_API_KEY]"),
        (re.compile(r"\bgsk_[A-Za-z0-9_\-]{20,}"), "[REDACTED_API_KEY]"),
        (re.compile(r"[\w.+\-]+@[\w\-]+\.[\w.\-]+"), "[REDACTED_EMAIL]"),
        (re.compile(r"(?<!\d)(?:\d[ \-]?){13,15}\d(?!\d)"), "[REDACTED_CREDIT_CARD]"),
        (re.compile(r"(?<!\d)\d{4}\s?\d{4}\s?\d{4}(?!\d)"), "[REDACTED_AADHAAR]"),
        (
            re.compile(r"(?<![\d+])(?:\+\d{1,3}[\s.\-]?)?(?:\d[\s.\-]?){9,12}\d(?!\d)"),
            "[REDACTED_PHONE]",
        ),
    ]

    @staticmethod
    def scrub_text(text: str) -> str:
        for pattern, tag in PIIScrubber._PATTERNS:
            text = pattern.sub(tag, text)
        return text
