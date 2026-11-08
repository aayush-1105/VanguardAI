import re


class PromptGuard:
    """Heuristic blocklist for obvious jailbreak phrasing."""

    _PATTERNS = [
        re.compile(
            r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+"
            r"(instructions|prompts|rules|directions)",
            re.IGNORECASE,
        ),
        re.compile(r"disregard\s+(all\s+)?(previous|prior|above)", re.IGNORECASE),
        re.compile(r"\bDAN\b"),
        re.compile(r"system\s+prompt", re.IGNORECASE),
        re.compile(r"\bbypass\b", re.IGNORECASE),
        re.compile(r"developer\s+mode", re.IGNORECASE),
        re.compile(r"jailbreak", re.IGNORECASE),
        re.compile(r"pretend\s+(you\s+are|to\s+be)", re.IGNORECASE),
    ]

    @staticmethod
    def check_injection(text: str) -> bool:
        return any(pattern.search(text) for pattern in PromptGuard._PATTERNS)
