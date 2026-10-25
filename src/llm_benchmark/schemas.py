import json
import re
from typing import Literal, Tuple, Optional, Dict, Any
from pydantic import BaseModel, ValidationError

AllowedCategory = Literal[
    "Billing",
    "Technical Bug",
    "Account Access",
    "Feature Request",
    "Security & Compliance",
    "General Inquiry",
]
AllowedUrgency = Literal["Low", "Medium", "High", "Critical"]
AllowedDepartment = Literal["Finance", "Tier-1 Support", "Engineering", "Security"]


class TicketPrediction(BaseModel):
    category: AllowedCategory
    urgency: AllowedUrgency
    department: AllowedDepartment


def parse_and_validate_output(raw_text: str) -> Tuple[bool, bool, Dict[str, str]]:
    """
    Returns:
      - is_strict_json: True if raw_text parses directly via json.loads() without wrapper chatter
      - is_schema_valid: True if keys and values strictly match Allowed literals (no hallucinations)
      - parsed_dict: Extracted fields (or 'INVALID' placeholders if parsing failed)
    """
    cleaned = raw_text.strip()
    is_strict_json = False
    parsed_data: Optional[Dict[str, Any]] = None

    # 1. Check strict raw JSON compliance
    try:
        parsed_data = json.loads(cleaned)
        if isinstance(parsed_data, dict):
            is_strict_json = True
    except Exception:
        # 2. Fallback: extract first {...} block if base model added markdown/chatter
        match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
        if match:
            try:
                parsed_data = json.loads(match.group(0))
            except Exception:
                parsed_data = None

    if not isinstance(parsed_data, dict):
        return False, False, {
            "category": "INVALID_JSON",
            "urgency": "INVALID_JSON",
            "department": "INVALID_JSON",
        }

    # 3. Validate against strict Pydantic schema (catches hallucinated labels)
    try:
        validated = TicketPrediction(**parsed_data)
        return is_strict_json, True, validated.model_dump()
    except ValidationError:
        return is_strict_json, False, {
            "category": str(parsed_data.get("category", "HALLUCINATED")),
            "urgency": str(parsed_data.get("urgency", "HALLUCINATED")),
            "department": str(parsed_data.get("department", "HALLUCINATED")),
        }