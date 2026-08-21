import re
from typing import Tuple

class PromptGuard:
    """
    Protects multi-agent reasoning from prompt injection, adversarial overrides,
    and corrupted payload attacks (Comp 16).
    """
    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
        r"system\s+prompt\s+override",
        r"you\s+are\s+now\s+in\s+developer\s+mode",
        r"disregard\s+safety\s+guidelines",
        r"<script.*?>",
        r"DROP\s+TABLE",
        r"SELECT\s+.*?\s+FROM"
    ]

    def __init__(self):
        self.regexes = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS]

    def validate_input(self, text: str) -> Tuple[bool, str]:
        if not text or len(text.strip()) == 0:
            return True, "Valid"
        
        if len(text) > 10000:
            return False, "Payload length exceeds security threshold (10,000 chars)"

        for regex in self.regexes:
            if regex.search(text):
                return False, f"Potential adversarial prompt injection detected: '{regex.pattern}'"

        return True, "Valid"
