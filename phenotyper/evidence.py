from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


@dataclass
class Evidence:
    patient_id: str
    note_id: str
    note_date: Optional[str]
    note_type: Optional[str]

    # Use "field" as the canonical name (schema-friendly),
    # but keep "entity" as an alias for backwards compatibility.
    field: str

    value: str
    start: int
    end: int
    snippet: str

    # medspaCy / TargetMatcher label (what pattern matched)
    label: str

    confidence: float = 0.70

    # ConText flags (for aggregation logic)
    is_negated: bool = False
    is_uncertain: bool = False

    @property
    def entity(self) -> str:
        """Backward-compatible alias used by older parts of the app/UI."""
        return self.field

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Include alias explicitly so your Dash table can show either column
        d["entity"] = self.field
        return d
