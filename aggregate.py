from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

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
    """Higher is better: (note type precedence, note date ordinal)."""
    tscore = NOTE_TYPE_PRECEDENCE.get(note_type, 0)
    d = _parse_date(note_date)
    dscore = int(d.toordinal()) if d else 0
    return tscore, dscore


def _confidence_bucket(note_type: Optional[str]) -> str:
    """Map raw note_type strings into stable v1 confidence buckets."""
    if not note_type:
        return "unknown"

    nt = str(note_type).lower()

    if "addendum" in nt:
        return "addendum"
    if "path" in nt:
        return "pathology"
    if "oncology" in nt or "consult" in nt:
        return "consult"
    if "radio" in nt:
        return "radiology"

    return "unknown"


def _is_missing(v: Any, *, treat_unknown_as_missing: bool = True) -> bool:
    """
    Return True for values that should not be selected as patient-level outputs.

    This prevents pandas NaN values from being treated as real extracted values,
    which can otherwise create bogus source provenance for blank fields.
    """
    if v is None:
        return True

    # pandas/numpy NaN usually reaches us as float("nan"), but keep this
    # dependency-light so aggregate.py does not need pandas.
    if isinstance(v, float) and math.isnan(v):
        return True

    s = str(v).strip()
    if s == "":
        return True

    if s.lower() in {"nan", "none", "null"}:
        return True

    if treat_unknown_as_missing and s.lower() == "unknown":
        return True

    return False


def _norm_val(v: Any) -> str:
    """
    Normalize values for evidence matching.

    This prevents int-vs-str mismatches, e.g. ki67 35 vs "35".
    Missing values normalize to the empty string.
    """
    if _is_missing(v, treat_unknown_as_missing=False):
        return ""

    if isinstance(v, float) and v.is_integer():
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

    Clean support means there is evidence for this patient+field+value that is
    not negated and not uncertain. Any support includes negated/uncertain rows.
    """
    if _is_missing(value):
        return False, False

    clean = False
    any_support = False
    target_val = _norm_val(value)

    for e in evidence_rows:
        if getattr(e, "patient_id", None) != patient_id:
            continue

        # Accept either e.field or legacy e.entity.
        ev_field = getattr(e, "field", None) or getattr(e, "entity", None)
        if ev_field != field:
            continue

        ev_val = _norm_val(getattr(e, "value", None))
        if ev_val != target_val:
            continue

        any_support = True
        is_neg = bool(getattr(e, "is_negated", False))
        is_unc = bool(getattr(e, "is_uncertain", False))

        if not is_neg and not is_unc:
            clean = True
            break

    return clean, any_support


def _compute_her2_final(
    ihc_score: Optional[str],
    fish: Optional[str],
    explicit_status: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Determine HER2 final status deterministically.

    Returns:
      (her2_final_status, her2_confidence)

    Rules:
      - FISH overrides IHC/explicit status.
      - IHC 3+ => Positive.
      - IHC 0 or 1+ => Negative.
      - IHC 2+ => Equivocal.
      - If no FISH/IHC is present, fall back to explicit Positive/Negative/Equivocal.
    """
    if not _is_missing(fish):
        f = str(fish).strip().lower()

        if "not" in f and "ampl" in f:
            return "Negative", "fish"
        if "ampl" in f:
            return "Positive", "fish"
        if "neg" in f:
            return "Negative", "fish"
        if "pos" in f:
            return "Positive", "fish"

        return "Unknown", "fish"

    if not _is_missing(ihc_score):
        s = str(ihc_score).strip().lower().replace(" ", "")

        if s.startswith("3"):
            return "Positive", "ihc"
        if s.startswith("2"):
            return "Equivocal", "ihc"
        if s.startswith("1") or s.startswith("0"):
            return "Negative", "ihc"

        return "Unknown", "ihc"

    if not _is_missing(explicit_status):
        status = str(explicit_status).strip()
        if status in {"Positive", "Negative", "Equivocal"}:
            return status, "explicit"

    return None, None


def aggregate_patient(note_rows: List[Dict[str, Any]], evidence_rows: List[Evidence]) -> Dict[str, Any]:
    """Aggregate multiple note-level phenotype dicts into a patient-level phenotype dict.

    Strategy, field-wise:
      1) Prefer values with at least one non-negated, non-uncertain evidence mention.
      2) Among those, prefer higher note_type precedence.
      3) If still tied, prefer newest date.
      4) If still tied, keep first encountered.
      5) If no clean evidence exists anywhere, fall back to first non-missing value.
    """
    if not note_rows:
        return {}

    patient_id = note_rows[0].get("patient_id")
    if _is_missing(patient_id):
        return {}

    patient_id = str(patient_id)

    # v1 base fields picked directly from note-level extraction.
    # HER2 final is included so explicit HER2 positive/negative mentions can survive
    # when there is no IHC/FISH score. A derived FISH/IHC final can still override it.
    base_fields = [
        "er_status",
        "er_percent",
        "pr_status",
        "pr_percent",
        "her2_ihc_score",
        "her2_fish",
        "her2_final_status",
        "ki67_percent",
        "histology",
        "grade",
        "stage_clinical",
        "stage_path",
        "metastatic_disease",
    ]

    out: Dict[str, Any] = {"patient_id": patient_id}

    # Initialize all v1 schema columns to None.
    for col in PHENOTYPE_COLUMNS_V1:
        if col != "patient_id":
            out[col] = None

    # Add source columns for transparent provenance.
    for f in base_fields:
        out[f"{f}__source_note_id"] = None
        out[f"{f}__source_note_type"] = None
        out[f"{f}__source_note_date"] = None

    # Sort notes by (type precedence, date) descending.
    ranked = sorted(
        note_rows,
        key=lambda r: _note_score(r.get("note_type"), r.get("note_date")),
        reverse=True,
    )

    for f in base_fields:
        chosen = None
        chosen_row = None

        # First pass: choose value with clean evidence.
        for r in ranked:
            v = r.get(f)
            if _is_missing(v):
                continue

            has_clean, _ = _evidence_support(evidence_rows, patient_id, f, v)
            if has_clean:
                chosen = v
                chosen_row = r
                break

        # Second pass: choose first non-missing value, even without evidence support.
        # This protects fields filled by note-level logic but not represented in evidence.
        if chosen is None:
            for r in ranked:
                v = r.get(f)
                if _is_missing(v):
                    continue

                chosen = v
                chosen_row = r
                break

        if chosen_row is None:
            continue

        out[f] = chosen
        out[f"{f}__source_note_id"] = chosen_row.get("note_id")
        out[f"{f}__source_note_type"] = chosen_row.get("note_type")
        out[f"{f}__source_note_date"] = chosen_row.get("note_date")

        bucket = _confidence_bucket(chosen_row.get("note_type"))

        if f == "er_status":
            out["er_confidence"] = bucket
        elif f == "er_percent":
            out["er_confidence"] = out.get("er_confidence") or bucket
        elif f == "pr_status":
            out["pr_confidence"] = bucket
        elif f == "pr_percent":
            out["pr_confidence"] = out.get("pr_confidence") or bucket
        elif f == "her2_final_status":
            out["her2_confidence"] = out.get("her2_confidence") or bucket
        elif f == "ki67_percent":
            out["ki67_confidence"] = bucket
        elif f == "histology":
            out["histology_confidence"] = bucket
        elif f == "metastatic_disease":
            out["metastatic_disease_confidence"] = bucket

    # Derived HER2 final status + confidence.
    # FISH/IHC should override explicit status if available.
    her2_final, her2_conf = _compute_her2_final(
        out.get("her2_ihc_score"),
        out.get("her2_fish"),
        out.get("her2_final_status"),
    )

    if her2_final is not None and not _is_missing(her2_final):
        out["her2_final_status"] = her2_final
    if her2_conf is not None and not _is_missing(her2_conf, treat_unknown_as_missing=False):
        out["her2_confidence"] = her2_conf

    return out
