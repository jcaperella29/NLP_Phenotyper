from __future__ import annotations

import re
from typing import Dict, Any, List, Optional, Tuple

from .nlp import build_nlp
from .preprocess import clean_text
from .evidence import Evidence
from .normalize import (
    normalize_percent,
    reconcile_her2,
    normalize_histology,
    normalize_grade,
    normalize_stage,
    normalize_status,
)


# ---------
# ER/PR specific percent extraction (ONLY if a percent sign exists)
# ---------
_ER_PCT = re.compile(r"\b(?:estrogen receptor|er)\b[^%\n]{0,60}\b(\d{1,3})\s*%", re.I)
_PR_PCT = re.compile(r"\b(?:progesterone receptor|pr)\b[^%\n]{0,60}\b(\d{1,3})\s*%", re.I)

# ER/PR status fallback (robust)
_ER_POS_FALLBACK = re.compile(r"\b(?:estrogen receptor|er)\b[^a-z0-9]{0,10}\b(?:positive|pos|\+)\b", re.I)
_ER_NEG_FALLBACK = re.compile(r"\b(?:estrogen receptor|er)\b[^a-z0-9]{0,10}\b(?:negative|neg|-)\b", re.I)
_PR_POS_FALLBACK = re.compile(r"\b(?:progesterone receptor|pr)\b[^a-z0-9]{0,10}\b(?:positive|pos|\+)\b", re.I)
_PR_NEG_FALLBACK = re.compile(r"\b(?:progesterone receptor|pr)\b[^a-z0-9]{0,10}\b(?:negative|neg|-)\b", re.I)


def _get_context_flags(doc, ent) -> Tuple[bool, bool]:
    """
    Try to read negation/uncertainty flags from medspaCy ConText.
    Returns: (is_negated, is_uncertain)
    """
    is_neg = False
    is_unc = False
    try:
        if hasattr(ent, "_") and hasattr(ent._, "is_negated"):
            is_neg = bool(ent._.is_negated)
        if hasattr(ent, "_") and hasattr(ent._, "is_uncertain"):
            is_unc = bool(ent._.is_uncertain)
    except Exception:
        pass
    return is_neg, is_unc


def _find_er_percent(window: str) -> Optional[int]:
    m = _ER_PCT.search(window or "")
    if not m:
        return None
    val = int(m.group(1))
    return val if 0 <= val <= 100 else None


def _find_pr_percent(window: str) -> Optional[int]:
    m = _PR_PCT.search(window or "")
    if not m:
        return None
    val = int(m.group(1))
    return val if 0 <= val <= 100 else None


def extract_note(
    note_text: str,
    *,
    patient_id: str,
    note_id: str,
    note_date: Optional[str] = None,
    note_type: Optional[str] = None,
    model_name: str = "en_core_web_sm",
) -> Tuple[Dict[str, Any], List[Evidence]]:
    """Extract phenotypes from a single note."""
    text = clean_text(note_text)
    nlp = build_nlp(model_name)
    doc = nlp(text)

    phenos: Dict[str, Any] = {
        "patient_id": patient_id,
        "note_id": note_id,
        "note_date": note_date,
        "note_type": note_type,

        "er_status": None,
        "er_percent": None,

        "pr_status": None,
        "pr_percent": None,

        "her2_status": None,
        "her2_ihc_score": None,
        "her2_fish": None,

        "ki67_percent": None,
        "histology": None,
        "grade": None,
        "stage_clinical": None,
        "stage_path": None,
    }

    # intermediate HER2 fields
    her2_explicit: Optional[str] = None
    her2_ihc: Optional[str] = None
    her2_fish: Optional[str] = None

    evidence: List[Evidence] = []

    def add_ev(field: str, value: str, ent, label: str, conf: float = 0.70):
        snippet = text[max(ent.start_char - 80, 0): min(ent.end_char + 80, len(text))]
        is_negated, is_uncertain = _get_context_flags(doc, ent)

        # Evidence schema can vary (entity vs field). We'll support both safely.
        base_kwargs = {
            "patient_id": patient_id,
            "note_id": note_id,
            "note_date": note_date,
            "note_type": note_type,
            "value": value,
            "start": ent.start_char,
            "end": ent.end_char,
            "snippet": snippet,
            "label": label,
            "confidence": conf,
            "is_negated": is_negated,
            "is_uncertain": is_uncertain,
        }

        try:
            evidence.append(Evidence(field=field, **base_kwargs))
            return
        except TypeError:
            pass

        try:
            evidence.append(Evidence(entity=field, **base_kwargs))
            return
        except TypeError:
            pass

        # last resort: try both keys present
        evidence.append(Evidence(field=field, entity=field, **base_kwargs))

    # Iterate TargetMatcher entities
    for ent in doc.ents:
        label = ent.label_
        span = ent.text

        # ----------------------------
        # ER / PR
        # ----------------------------
        if label == "ER_POS":
            phenos["er_status"] = normalize_status("Positive")
            add_ev("er_status", "Positive", ent, label, 0.85)

            pct = _find_er_percent(ent.sent.text)
            if pct is not None:
                phenos["er_percent"] = pct
                add_ev("er_percent", str(pct), ent, label, 0.70)

        elif label == "ER_NEG":
            phenos["er_status"] = normalize_status("Negative")
            add_ev("er_status", "Negative", ent, label, 0.85)

            pct = _find_er_percent(ent.sent.text)
            if pct is not None:
                phenos["er_percent"] = pct
                add_ev("er_percent", str(pct), ent, label, 0.70)

        elif label == "PR_POS":
            phenos["pr_status"] = normalize_status("Positive")
            add_ev("pr_status", "Positive", ent, label, 0.85)

            pct = _find_pr_percent(ent.sent.text)
            if pct is not None:
                phenos["pr_percent"] = pct
                add_ev("pr_percent", str(pct), ent, label, 0.70)

        elif label == "PR_NEG":
            phenos["pr_status"] = normalize_status("Negative")
            add_ev("pr_status", "Negative", ent, label, 0.85)

            pct = _find_pr_percent(ent.sent.text)
            if pct is not None:
                phenos["pr_percent"] = pct
                add_ev("pr_percent", str(pct), ent, label, 0.70)

        # ----------------------------
        # HER2
        # ----------------------------
        elif label == "HER2_IHC":
            her2_ihc = span.strip()
            add_ev("her2_ihc_score", her2_ihc, ent, label, 0.80)

        elif label == "HER2_POS":
            her2_explicit = "Positive"
            add_ev("her2_status", "Positive", ent, label, 0.75)

        elif label == "HER2_NEG":
            her2_explicit = "Negative"
            add_ev("her2_status", "Negative", ent, label, 0.75)

        elif label == "HER2_FISH_POS":
            her2_fish = "Amplified"
            add_ev("her2_fish", "Amplified", ent, label, 0.85)

        elif label == "HER2_FISH_NEG":
            her2_fish = "Not amplified"
            add_ev("her2_fish", "Not amplified", ent, label, 0.85)

        # ----------------------------
        # Ki-67
        # ----------------------------
        elif label == "KI67":
            val = normalize_percent(span)
            if val is not None:
                phenos["ki67_percent"] = val
                add_ev("ki67_percent", str(val), ent, label, 0.80)

        # ----------------------------
        # Histology
        # ----------------------------
        elif label.startswith("HISTOLOGY_"):
            hist = normalize_histology(span if label != "HISTOLOGY_TEXT" else ent.sent.text)
            if hist:
                phenos["histology"] = hist
                add_ev("histology", hist, ent, label, 0.75)

        # ----------------------------
        # Grade
        # ----------------------------
        elif label == "GRADE":
            g = normalize_grade(ent.sent.text)
            if g:
                phenos["grade"] = g
                add_ev("grade", g, ent, label, 0.75)

        # ----------------------------
        # Stage
        # ----------------------------
        elif label in {"STAGE_PATH", "STAGE_CLIN", "STAGE_GENERIC"}:
            st = normalize_stage(ent.text if label != "STAGE_GENERIC" else ent.sent.text)
            if st:
                if label == "STAGE_PATH":
                    phenos["stage_path"] = st
                    add_ev("stage_path", st, ent, label, 0.75)
                elif label == "STAGE_CLIN":
                    phenos["stage_clinical"] = st
                    add_ev("stage_clinical", st, ent, label, 0.75)
                else:
                    if not phenos.get("stage_path"):
                        phenos["stage_path"] = st
                        add_ev("stage_path", st, ent, label, 0.60)
                    elif not phenos.get("stage_clinical"):
                        phenos["stage_clinical"] = st
                        add_ev("stage_clinical", st, ent, label, 0.60)

    # ----------------------------
    # Regex fallback to GUARANTEE ER/PR status if present in text
    # ----------------------------
    if phenos["er_status"] is None:
        if _ER_POS_FALLBACK.search(text):
            phenos["er_status"] = normalize_status("Positive")
        elif _ER_NEG_FALLBACK.search(text):
            phenos["er_status"] = normalize_status("Negative")

    if phenos["pr_status"] is None:
        if _PR_POS_FALLBACK.search(text):
            phenos["pr_status"] = normalize_status("Positive")
        elif _PR_NEG_FALLBACK.search(text):
            phenos["pr_status"] = normalize_status("Negative")

    # If percent exists anywhere, try to fill missing percent (requires % sign)
    if phenos["er_percent"] is None:
        p = _find_er_percent(text)
        if p is not None:
            phenos["er_percent"] = p

    if phenos["pr_percent"] is None:
        p = _find_pr_percent(text)
        if p is not None:
            phenos["pr_percent"] = p

    # ----------------------------
    # reconcile HER2
    # ----------------------------
    her2_status, her2_ihc_score, her2_fish_norm = reconcile_her2(her2_ihc, her2_fish, her2_explicit)
    phenos["her2_status"] = normalize_status(her2_status)
    phenos["her2_ihc_score"] = her2_ihc_score
    phenos["her2_fish"] = her2_fish_norm

    return phenos, evidence
