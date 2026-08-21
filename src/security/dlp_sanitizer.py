import re
from typing import Dict, Any, Tuple

class CloudDLPSanitizer:
    """
    Sanitizes proprietary cleanroom recipe codes, internal chamber serials,
    and operator credentials from inspection payloads before LLM ingestion (Comp 15).
    """
    def __init__(self):
        # Proprietary cleanroom recipe pattern: RECIPE-XXX-1234
        self.recipe_pattern = re.compile(r"RECIPE-[A-Z0-9]+-[0-9]+", re.IGNORECASE)
        # Internal hardware serial: SN-[A-Z0-9]{6,12}
        self.serial_pattern = re.compile(r"SN-[A-Z0-9]{6,12}", re.IGNORECASE)
        # Operator email / PII
        self.email_pattern = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

    def sanitize_text(self, text: str) -> Tuple[str, Dict[str, int]]:
        redaction_counts = {"recipes": 0, "serials": 0, "emails": 0}
        
        def replace_recipe(match):
            redaction_counts["recipes"] += 1
            return "[REDACTED_RECIPE_IP]"

        def replace_serial(match):
            redaction_counts["serials"] += 1
            return "[REDACTED_HARDWARE_SN]"

        def replace_email(match):
            redaction_counts["emails"] += 1
            return "[REDACTED_OPERATOR_EMAIL]"

        sanitized = self.recipe_pattern.sub(replace_recipe, text)
        sanitized = self.serial_pattern.sub(replace_serial, sanitized)
        sanitized = self.email_pattern.sub(replace_email, sanitized)
        
        return sanitized, redaction_counts

    def sanitize_dict(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, int]]:
        total_counts = {"recipes": 0, "serials": 0, "emails": 0}
        cleaned_data = {}
        for k, v in data.items():
            if isinstance(v, str):
                cleaned_val, counts = self.sanitize_text(v)
                cleaned_data[k] = cleaned_val
                for ck, cv in counts.items():
                    total_counts[ck] += cv
            elif isinstance(v, dict):
                cleaned_val, counts = self.sanitize_dict(v)
                cleaned_data[k] = cleaned_val
                for ck, cv in counts.items():
                    total_counts[ck] += cv
            else:
                cleaned_data[k] = v
        return cleaned_data, total_counts
