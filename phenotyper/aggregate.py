from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
from dateutil.parser import parse as dtparse

from .schema import NOTE_TYPE_PRECEDENCE, PHENOTYPE_COLUMNS_V1
from .evidence import Evidence


def _parse_date(d: Optional[str]):
    if not d:
        return None
    try:
        return dtparse(d).date()
    except Exception:
        return None


def _note_score(note_type: Optional[str], note_date: Optional[str]) -> Tuple[int, int]:
    # Higher is better: (type precedence, date ordinal)
    tscore = NOTE_TYPE_PRECEDENCE.get(note_type, 0)
    d = _parse_date(note_date)
    dscore = int(d.toordinal()) if d else 0
    return tscore, dscore


def _confidence_bucket(note_type: Optional[str]) -> str:
    """
    Map raw note_type strings into stable v1 confidence buckets.
    """
    if not note_type:
        return "unknown"
    nt = str(note_type).lower()

    if "addendum" in nt:
        return "addendum"
    if "path" in nt:  # Pathology / SurgicalPathology
        return "pathology"
    if "oncology" in nt or "consult" in nt:
        return "consult"
    if "radio" in nt:
        return "radiology"

    return "unknown"


def _norm_val(v: Any) -> str:
    """
    Normalize values for evidence matching.
    This prevents int vs str mismatches (e.g., ki67 35 vs "35").
    """
    if v is None:
        return ""
    if isinstance(v, float):
        # avoid "35.0" when value is conceptually integer
        if v.is_integer():
            return str(int(v))
    return str(v).strip()


def _evidence_support(
    evidence_rows: List[Evidence],
    patient_id: str,
    field: str,
    value: Any,
) -> Tuple[bool, bool]:
    """
    Returns:
      (has_clean_support, has_any_support)

    'clean support' means there exists evidence for this patient+field+value
    that is NOT negated and NOT uncertain.

    'any support' means any evidence exists (even negated/uncertain).
    """
    clean = False
    any_support = False

    target_val = _norm_val(value)

    for e in evidence_rows:
        if getattr(e, "patient_id", None) != patient_id:
            continue

        # Accept either e.field or legacy e.entity
        ev_field = getattr(e, "field", None) or getattr(e, "entity", None)
        if ev_field != field:
            continue

        ev_val = _norm_val(getattr(e, "value", None))
        if ev_val != target_val:
            continue

        any_support = True
        is_neg = bool(getattr(e, "is_negated", False))
        is_unc = bool(getattr(e, "is_uncertain", False))

        if (not is_neg) and (not is_unc):
            clean = True
            break

    return clean, any_support


def _compute_her2_final(ihc_score: Optional[str], fish: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Determine HER2 final status deterministically.

    Returns: (her2_final_status, her2_confidence)

    Rules:
      - If FISH present: Amplified => Positive, Not amplified => Negative, else Unknown
      - Else use IHC:
          3+ => Positive
          0 or 1+ => Negative
          2+ => Equivocal
    """
    if fish:
        f = str(fish).strip().lower()
        if "ampl" in f and "not" not in f:
            return "Positive", "fish"
        if "not" in f and "ampl" in f:
            return "Negative", "fish"
        if "neg" in f:
            return "Negative", "fish"
        if "pos" in f:
            return "Positive", "fish"
        return "Unknown", "fish"

    if ihc_score:
        s = str(ihc_score).strip().lower()
        # accept "3+", "3 +", "score 3+" etc.
        if "3" in s and "+" in s:
            return "Positive", "ihc"
        if s.startswith("3"):
            return "Positive", "ihc"
        if s.startswith("2") and "+" in s:
            return "Equivocal", "ihc"
        if s.startswith("1") and "+" in s:
            return "Negative", "ihc"
        if s.startswith("0"):
            return "Negative", "ihc"
        # sometimes ihc appears as "0" "1" "2" "3"
        if s == "3":
            return "Positive", "ihc"
        if s == "2":
            return "Equivocal", "ihc"
        if s == "1" or s == "0":
            return "Negative", "ihc"
        return "Unknown", "ihc"

    return None, None


def aggregate_patient(note_rows: List[Dict[str, Any]], evidence_rows: List[Evidence]) -> Dict[str, Any]:
    """Aggregate multiple note-level phenotype dicts into a patient-level phenotype dict.

    Strategy (field-wise):
      1) Prefer values with at least one NON-negated, NON-uncertain evidence mention.
      2) Among those, prefer higher note_type precedence (pathology/addendum > consult > radiology).
      3) If still tied, prefer newest date.
      4) If still tied, keep first encountered.
      5) If no clean evidence exists anywhere, fall back to any non-empty value.
    """
    if not note_rows:
        return {}

    patient_id = note_rows[0].get("patient_id")
    if not patient_id:
        return {}

    # v1 "base" fields we pick directly from notes
    base_fields = [
        "er_status", "pr_status",
        "her2_ihc_score", "her2_fish",
        "ki67_percent", "histology", "grade",
        "stage_clinical", "stage_path",
        # optional: if your note extraction already includes these
        "er_percent", "pr_percent",
    ]

    out: Dict[str, Any] = {"patient_id": patient_id}

    # Initialize all v1 schema columns to None (except patient_id)
    for col in PHENOTYPE_COLUMNS_V1:
        if col != "patient_id":
            out[col] = None

    # for transparency, keep sources for base fields
    for f in base_fields:
        out[f"{f}__source_note_id"] = None
        out[f"{f}__source_note_type"] = None
        out[f"{f}__source_note_date"] = None

    # Sort notes by (type precedence, date) descending
    ranked = sorted(
        note_rows,
        key=lambda r: _note_score(r.get("note_type"), r.get("note_date")),
        reverse=True,
    )

    # Pick base fields
    for f in base_fields:
        chosen = None
        chosen_row = None

        # First pass: choose value with clean evidence
        for r in ranked:
            v = r.get(f)
            if v is None or v == "":
                continue

            has_clean, _ = _evidence_support(evidence_rows, patient_id, f, v)
            if has_clean:
                chosen = v
                chosen_row = r
                break

        # Second pass fallback: first non-empty value
        if chosen is None:
            for r in ranked:
                v = r.get(f)
                if v is None or v == "":
                    continue
                chosen = v
                chosen_row = r
                break

        if chosen_row is not None:
            out[f] = chosen
            out[f"{f}__source_note_id"] = chosen_row.get("note_id")
            out[f"{f}__source_note_type"] = chosen_row.get("note_type")
            out[f"{f}__source_note_date"] = chosen_row.get("note_date")

            # Populate confidence buckets for selected fields
            bucket = _confidence_bucket(chosen_row.get("note_type"))
            if f == "er_status":
                out["er_confidence"] = bucket
            elif f == "er_percent":
                out["er_confidence"] = out.get("er_confidence") or bucket
            elif f == "pr_status":
                out["pr_confidence"] = bucket
            elif f == "pr_percent":
                out["pr_confidence"] = out.get("pr_confidence") or bucket
            elif f == "ki67_percent":
                out["ki67_confidence"] = bucket
            elif f == "histology":
                out["histology_confidence"] = bucket

    # Derived HER2 final status + confidence
    her2_final, her2_conf = _compute_her2_final(out.get("her2_ihc_score"), out.get("her2_fish"))
    if her2_final is not None:
        out["her2_final_status"] = her2_final
    if her2_conf is not None:
        out["her2_confidence"] = her2_conf

    return out
