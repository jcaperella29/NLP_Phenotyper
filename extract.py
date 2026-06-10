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
_ER_PCT = re.compile(r"\b(?:estrogen\s+receptor|er)\b[^%\n]{0,80}\b(\d{1,3})\s*%", re.I)
_PR_PCT = re.compile(r"\b(?:progesterone\s+receptor|pr)\b[^%\n]{0,80}\b(\d{1,3})\s*%", re.I)

# ER/PR status fallback.
# Handles:
#   ER: Positive
#   ER positive
#   Estrogen receptor (ER): Positive (90%)
#   Progesterone receptor (PR): Positive (80%)
_ER_STATUS_FALLBACK = re.compile(
    r"\b(?:estrogen\s+receptor\s*(?:\(\s*ER\s*\))?|ER)\b"
    r"\s*(?:\)|\])?\s*(?:status)?\s*(?:is|:|-)?\s*"
    r"(positive|pos|negative|neg|\+|-)",
    re.I,
)
_PR_STATUS_FALLBACK = re.compile(
    r"\b(?:progesterone\s+receptor\s*(?:\(\s*PR\s*\))?|PR)\b"
    r"\s*(?:\)|\])?\s*(?:status)?\s*(?:is|:|-)?\s*"
    r"(positive|pos|negative|neg|\+|-)",
    re.I,
)

# Backwards-compatible names if older code imports these.
_ER_POS_FALLBACK = re.compile(r"\b(?:estrogen receptor|er)\b[^a-z0-9]{0,10}\b(?:positive|pos|\+)\b", re.I)
_ER_NEG_FALLBACK = re.compile(r"\b(?:estrogen receptor|er)\b[^a-z0-9]{0,10}\b(?:negative|neg|-)\b", re.I)
_PR_POS_FALLBACK = re.compile(r"\b(?:progesterone receptor|pr)\b[^a-z0-9]{0,10}\b(?:positive|pos|\+)\b", re.I)
_PR_NEG_FALLBACK = re.compile(r"\b(?:progesterone receptor|pr)\b[^a-z0-9]{0,10}\b(?:negative|neg|-)\b", re.I)


# HER2 fallback.
# Handles:
#   HER2: 2+ (equivocal)
#   HER2 3+
#   HER2: Negative (0)
#   HER2 is negative by immunohistochemistry
#   HER2 status is pending confirmatory testing
_HER2_SCORE_FALLBACK = re.compile(
    r"\bHER-?2\b(?:\s+IHC|\s+immunohistochemistry)?\s*[:\-]?\s*"
    r"([0-3]\s*\+?|0)\s*(?:\((positive|negative|equivocal)\))?",
    re.I,
)
_HER2_STATUS_FALLBACK = re.compile(
    r"\bHER-?2\b(?:\s+status)?\s*(?:is|:|-)?\s*"
    r"(positive|negative|equivocal|pending)\b"
    r"(?:[^\n]{0,30}?\(?\s*([0-3]\s*\+?|0)\s*\)?)?",
    re.I,
)


# Metastatic disease fallback.
# Normalized patient-level values:
#   Present | Absent | Unknown
# Evidence rows use positive clinical meaning; e.g. "No evidence of metastatic disease"
# becomes metastatic_disease = Absent rather than a negated Present.
_METASTASIS_ABSENT_FALLBACK = re.compile(
    r"\b(?:no\s+(?:clinical\s+)?evidence\s+of|without\s+(?:evidence\s+of)?|negative\s+for)\s+"
    r"(?:distant\s+)?(?:metastatic\s+disease|metastasis|metastases)\b",
    re.I,
)
_METASTASIS_UNCERTAIN_FALLBACK = re.compile(
    r"\b(?:possible|possibly|suspicious\s+for|cannot\s+exclude|may\s+represent|rule\s+out)\b"
    r"[^\n.]{0,80}\b(?:metastatic\s+disease|metastasis|metastases)\b|"
    r"\b(?:metastatic\s+disease|metastasis|metastases)\b[^\n.]{0,80}"
    r"\b(?:possible|possibly|cannot\s+exclude|not\s+excluded|suspected)\b",
    re.I,
)
_METASTASIS_PRESENT_FALLBACK = re.compile(
    r"\b(?:metastatic\s+breast\s+cancer|metastatic\s+disease|distant\s+metastasis|"
    r"(?:bone|liver|lung|brain|distant)\s+metastases|metastasis\s+to\s+(?:bone|liver|lung|brain))\b",
    re.I,
)
_FAMILY_HISTORY_WINDOW = re.compile(
    r"\b(?:family\s+history|mother|father|sister|brother|aunt|uncle|grandmother|grandfather)\b",
    re.I,
)


# Robust regex fallbacks for common pathology/oncology note formats.
_KI67_FALLBACK = re.compile(r"\bki[-\s]?67\b[^0-9%]{0,20}(\d{1,3})\s*%", re.I)
_GRADE_FALLBACK = re.compile(
    r"\b(?:nottingham\s+)?grade\b[^0-9]{0,15}([1-3])\b|\bgrade\s+(?:I{1,3}|1|2|3)\b",
    re.I,
)
_STAGE_PATH_TNM_FALLBACK = re.compile(
    r"\bpT[0-4][a-c]?(?:\s*pN[0-3][a-c]?)?(?:\s*pM[01])?\b",
    re.I,
)
_STAGE_CLIN_TNM_FALLBACK = re.compile(
    r"\bcT[0-4][a-c]?(?:\s*cN[0-3][a-c]?)?(?:\s*cM[01])?\b",
    re.I,
)
_STAGE_CLIN_ROMAN_FALLBACK = re.compile(
    r"\bclinical\s+stage\s*(I|II|III|IV)[ABC]?\b",
    re.I,
)
_STAGE_PATH_ROMAN_FALLBACK = re.compile(
    r"\bpathologic(?:al)?\s+stage\s*(I|II|III|IV)[ABC]?\b",
    re.I,
)
_STAGE_ROMAN_FALLBACK = re.compile(
    r"\b(?:pathologic(?:al)?|clinical)?\s*stage\s*(I|II|III|IV)[ABC]?\b",
    re.I,
)


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


def _clean_ihc_score(score: Optional[str]) -> Optional[str]:
    if not score:
        return None
    return re.sub(r"\s+", "", str(score).strip())


def _has_family_history_context(text: str, start: int, end: int) -> bool:
    """Return True if a match appears to be about family history rather than the patient."""
    window = text[max(start - 80, 0): min(end + 40, len(text))]
    return bool(_FAMILY_HISTORY_WINDOW.search(window))


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

        "metastatic_disease": None,
    }

    # intermediate HER2 fields
    her2_explicit: Optional[str] = None
    her2_ihc: Optional[str] = None
    her2_fish: Optional[str] = None

    evidence: List[Evidence] = []

    def add_ev(field: str, value: str, ent, label: str, conf: float = 0.70):
        snippet = text[max(ent.start_char - 80, 0): min(ent.end_char + 80, len(text))]
        is_negated, is_uncertain = _get_context_flags(doc, ent)

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

        evidence.append(Evidence(field=field, entity=field, **base_kwargs))

    def add_regex_ev(
        field: str,
        value: str,
        match,
        label: str,
        conf: float = 0.65,
        *,
        is_negated: bool = False,
        is_uncertain: bool = False,
    ):
        snippet = text[max(match.start() - 80, 0): min(match.end() + 80, len(text))]
        evidence.append(
            Evidence(
                patient_id=patient_id,
                note_id=note_id,
                note_date=note_date,
                note_type=note_type,
                field=field,
                value=str(value),
                start=match.start(),
                end=match.end(),
                snippet=snippet,
                label=label,
                confidence=conf,
                is_negated=is_negated,
                is_uncertain=is_uncertain,
            )
        )

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
            her2_ihc = _clean_ihc_score(span.strip())
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
                    sent_text = ent.sent.text.lower()
                    if "clinical" in sent_text:
                        if not phenos.get("stage_clinical"):
                            phenos["stage_clinical"] = st
                            add_ev("stage_clinical", st, ent, label, 0.60)
                    elif "pathologic" in sent_text or "pathological" in sent_text:
                        if not phenos.get("stage_path"):
                            phenos["stage_path"] = st
                            add_ev("stage_path", st, ent, label, 0.60)
                    else:
                        if not phenos.get("stage_path"):
                            phenos["stage_path"] = st
                            add_ev("stage_path", st, ent, label, 0.60)
                        elif not phenos.get("stage_clinical"):
                            phenos["stage_clinical"] = st
                            add_ev("stage_clinical", st, ent, label, 0.60)

    # ----------------------------
    # Regex fallback to GUARANTEE ER/PR status if present in text,
    # while also adding evidence rows for auditability.
    # ----------------------------
    if phenos["er_status"] is None:
        m = _ER_STATUS_FALLBACK.search(text)
        if m:
            status = normalize_status(m.group(1))
            if status in {"Positive", "Negative"}:
                phenos["er_status"] = status
                add_regex_ev("er_status", status, m, "ER_STATUS_REGEX", 0.82)

    if phenos["pr_status"] is None:
        m = _PR_STATUS_FALLBACK.search(text)
        if m:
            status = normalize_status(m.group(1))
            if status in {"Positive", "Negative"}:
                phenos["pr_status"] = status
                add_regex_ev("pr_status", status, m, "PR_STATUS_REGEX", 0.82)

    # If percent exists anywhere, try to fill missing percent and add evidence.
    if phenos["er_percent"] is None:
        m = _ER_PCT.search(text)
        if m:
            val = int(m.group(1))
            if 0 <= val <= 100:
                phenos["er_percent"] = val
                add_regex_ev("er_percent", str(val), m, "ER_PERCENT_REGEX", 0.72)

    if phenos["pr_percent"] is None:
        m = _PR_PCT.search(text)
        if m:
            val = int(m.group(1))
            if 0 <= val <= 100:
                phenos["pr_percent"] = val
                add_regex_ev("pr_percent", str(val), m, "PR_PERCENT_REGEX", 0.72)

    # ----------------------------
    # HER2 regex fallback rescue.
    # This captures common lab formats that TargetMatcher can miss.
    # ----------------------------
    if her2_ihc is None:
        m = _HER2_SCORE_FALLBACK.search(text)
        if m:
            score = _clean_ihc_score(m.group(1))
            if score:
                her2_ihc = score
                add_regex_ev("her2_ihc_score", score, m, "HER2_IHC_SCORE_REGEX", 0.78)

                score_status, _, _ = reconcile_her2(score, None, m.group(2))
                score_status = normalize_status(score_status)
                if score_status in {"Positive", "Negative", "Equivocal"}:
                    her2_explicit = her2_explicit or score_status
                    add_regex_ev("her2_status", score_status, m, "HER2_STATUS_FROM_IHC_REGEX", 0.74)

    # Status fallback is useful even when no numeric IHC score is present.
    m = _HER2_STATUS_FALLBACK.search(text)
    if m:
        raw_status = m.group(1)
        score = _clean_ihc_score(m.group(2)) if m.group(2) else None

        if score and her2_ihc is None:
            her2_ihc = score
            add_regex_ev("her2_ihc_score", score, m, "HER2_IHC_SCORE_REGEX", 0.78)

        if raw_status.lower() == "pending":
            # Keep pending as uncertain evidence, but do not let it override
            # a real positive/negative/equivocal value.
            add_regex_ev(
                "her2_status",
                "Unknown",
                m,
                "HER2_PENDING_REGEX",
                0.60,
                is_uncertain=True,
            )
        elif her2_explicit is None:
            status = normalize_status(raw_status)
            if status in {"Positive", "Negative", "Equivocal"}:
                her2_explicit = status
                add_regex_ev("her2_status", status, m, "HER2_STATUS_REGEX", 0.76)

    # ----------------------------
    # Regex fallback rescue for Ki-67 / grade / stage
    # ----------------------------
    if phenos["ki67_percent"] is None:
        m = _KI67_FALLBACK.search(text)
        if m:
            val = int(m.group(1))
            if 0 <= val <= 100:
                phenos["ki67_percent"] = val
                add_regex_ev("ki67_percent", str(val), m, "KI67_REGEX", 0.78)

    if phenos["grade"] is None:
        m = _GRADE_FALLBACK.search(text)
        if m:
            raw = m.group(0)
            g = normalize_grade(raw)
            if g is None and m.group(1):
                g = m.group(1)
            if g:
                phenos["grade"] = g
                add_regex_ev("grade", str(g), m, "GRADE_REGEX", 0.72)

    if phenos["stage_path"] is None:
        m = _STAGE_PATH_TNM_FALLBACK.search(text)
        if m:
            phenos["stage_path"] = m.group(0)
            add_regex_ev("stage_path", m.group(0), m, "STAGE_PATH_TNM_REGEX", 0.72)

    if phenos["stage_clinical"] is None:
        m = _STAGE_CLIN_TNM_FALLBACK.search(text)
        if m:
            phenos["stage_clinical"] = m.group(0)
            add_regex_ev("stage_clinical", m.group(0), m, "STAGE_CLIN_TNM_REGEX", 0.72)

    # Explicit Roman clinical/pathologic stage should be handled independently.
    # This fixes cases like:
    #   Clinical stage I.
    # even when a pathologic TNM stage is also present in another note.
    if phenos["stage_clinical"] is None:
        m = _STAGE_CLIN_ROMAN_FALLBACK.search(text)
        if m:
            st = normalize_stage(m.group(0)) or m.group(1).upper()
            phenos["stage_clinical"] = st
            add_regex_ev("stage_clinical", st, m, "STAGE_CLIN_REGEX", 0.72)

    if phenos["stage_path"] is None:
        m = _STAGE_PATH_ROMAN_FALLBACK.search(text)
        if m:
            st = normalize_stage(m.group(0)) or m.group(1).upper()
            phenos["stage_path"] = st
            add_regex_ev("stage_path", st, m, "STAGE_PATH_REGEX", 0.72)

    # Generic Roman stage fallback. If explicitly clinical/pathologic, respect that;
    # otherwise fill whichever stage slot is still empty.
    m = _STAGE_ROMAN_FALLBACK.search(text)
    if m:
        st = normalize_stage(m.group(0)) or m.group(1).upper()
        lowered = m.group(0).lower()

        if "clinical" in lowered:
            if phenos["stage_clinical"] is None:
                phenos["stage_clinical"] = st
                add_regex_ev("stage_clinical", st, m, "STAGE_CLIN_REGEX", 0.70)

        elif "pathologic" in lowered or "pathological" in lowered:
            if phenos["stage_path"] is None:
                phenos["stage_path"] = st
                add_regex_ev("stage_path", st, m, "STAGE_PATH_REGEX", 0.70)

        else:
            if phenos["stage_path"] is None:
                phenos["stage_path"] = st
                add_regex_ev("stage_path", st, m, "STAGE_GENERIC_REGEX", 0.60)
            elif phenos["stage_clinical"] is None:
                phenos["stage_clinical"] = st
                add_regex_ev("stage_clinical", st, m, "STAGE_GENERIC_REGEX", 0.60)

    # ----------------------------
    # Metastatic disease fallback.
    # Order matters:
    #   1) explicit absence
    #   2) uncertain/hypothetical mentions as uncertain Unknown evidence only
    #   3) current/patient-level present mentions
    # Family-history contexts are kept out of patient-level positive calls.
    # ----------------------------
    if phenos["metastatic_disease"] is None:
        m = _METASTASIS_ABSENT_FALLBACK.search(text)
        if m and not _has_family_history_context(text, m.start(), m.end()):
            phenos["metastatic_disease"] = "Absent"
            add_regex_ev("metastatic_disease", "Absent", m, "METASTASIS_ABSENT_REGEX", 0.78)

    if phenos["metastatic_disease"] is None:
        m = _METASTASIS_UNCERTAIN_FALLBACK.search(text)
        if m and not _has_family_history_context(text, m.start(), m.end()):
            add_regex_ev(
                "metastatic_disease",
                "Unknown",
                m,
                "METASTASIS_UNCERTAIN_REGEX",
                0.58,
                is_uncertain=True,
            )

    if phenos["metastatic_disease"] is None:
        m = _METASTASIS_PRESENT_FALLBACK.search(text)
        if m and not _has_family_history_context(text, m.start(), m.end()):
            # Avoid converting explicitly negated/uncertain snippets into Present.
            local = text[max(m.start() - 40, 0): min(m.end() + 40, len(text))].lower()
            blocked = any(
                phrase in local
                for phrase in [
                    "no evidence of",
                    "without evidence",
                    "negative for",
                    "cannot exclude",
                    "not excluded",
                    "possible",
                    "possibly",
                    "suspicious for",
                    "rule out",
                    "family history",
                ]
            )
            if not blocked:
                phenos["metastatic_disease"] = "Present"
                add_regex_ev("metastatic_disease", "Present", m, "METASTASIS_PRESENT_REGEX", 0.76)


    # ----------------------------
    # reconcile HER2
    # ----------------------------
    her2_status, her2_ihc_score, her2_fish_norm = reconcile_her2(her2_ihc, her2_fish, her2_explicit)
    phenos["her2_status"] = normalize_status(her2_status)
    phenos["her2_ihc_score"] = her2_ihc_score
    phenos["her2_fish"] = her2_fish_norm

    return phenos, evidence
