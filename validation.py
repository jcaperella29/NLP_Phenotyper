from __future__ import annotations

import math
import re
from typing import Iterable, List, Tuple, Any, Optional

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


DEFAULT_REVIEW_FIELDS = [
    "er_status",
    "pr_status",
    "her2_final_status",
    "metastatic_disease",
    "stage_path",
    "stage_clinical",
    "ki67_percent",
    "histology",
    "grade",
]


def add_review_columns(patient_df: pd.DataFrame, fields: Iterable[str] = DEFAULT_REVIEW_FIELDS) -> pd.DataFrame:
    """
    Adds reviewer-friendly columns without overwriting model outputs.

    For each field:
    - field__reviewed_value
    - field__review_status
    - field__reviewer_note
    """
    out = patient_df.copy()
    for field in fields:
        if field not in out.columns:
            continue
        for suffix, default in [
            ("__reviewed_value", ""),
            ("__review_status", "unreviewed"),
            ("__reviewer_note", ""),
        ]:
            col = f"{field}{suffix}"
            if col not in out.columns:
                out[col] = default
    return out


def _is_missing(v: Any, *, treat_unknown_as_missing: bool = True) -> bool:
    """Return True for blank/null/NaN values used in prediction/gold files."""
    if v is None:
        return True

    if isinstance(v, float) and math.isnan(v):
        return True

    s = str(v).strip()
    if s == "":
        return True

    if s.lower() in {"nan", "none", "null", "na", "n/a"}:
        return True

    if treat_unknown_as_missing and s.lower() == "unknown":
        return True

    return False


def _norm_simple(v: Any) -> Optional[str]:
    """Normalize generic categorical/string values for validation comparison."""
    if _is_missing(v):
        return None

    s = str(v).strip()
    s = re.sub(r"\s+", " ", s)

    # Normalize common integer-like floats from CSV roundtrips: 18.0 -> 18
    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]

    return s


def _norm_status(v: Any) -> Optional[str]:
    """Normalize Positive/Negative/Equivocal/Present/Absent style values."""
    s = _norm_simple(v)
    if s is None:
        return None

    low = s.lower()

    status_map = {
        "positive": "Positive",
        "pos": "Positive",
        "+": "Positive",
        "negative": "Negative",
        "neg": "Negative",
        "-": "Negative",
        "equivocal": "Equivocal",
        "borderline": "Equivocal",
        "indeterminate": "Equivocal",
        "present": "Present",
        "yes": "Present",
        "detected": "Present",
        "absent": "Absent",
        "no": "Absent",
        "not present": "Absent",
        "not detected": "Absent",
        "none detected": "Absent",
        "historical": "Historical",
        "history": "Historical",
    }

    return status_map.get(low, s)


def _roman_to_int(roman: str) -> Optional[int]:
    roman = roman.upper().strip()
    mapping = {"I": 1, "II": 2, "III": 3, "IV": 4}
    return mapping.get(roman)


def _norm_stage(v: Any) -> Optional[str]:
    """
    Normalize stage values for validation comparison.

    Examples:
    - "Clinical stage I" -> "I"
    - "Stage IA" -> "I"
    - "stage 1" -> "I"
    - "pT1c pN0" -> "PT1C PN0"
    """
    s = _norm_simple(v)
    if s is None:
        return None

    raw = s.strip()
    low = raw.lower()

    # TNM strings: keep details, but normalize case/spaces.
    if re.search(r"\b[pc]?t[0-4]", low):
        compact = re.sub(r"\s+", " ", raw.upper()).strip()
        return compact

    m = re.search(r"\bstage\s*([ivx]+|[1-4])\s*([abc])?\b", low, re.I)
    if m:
        core = m.group(1).upper()
        if core.isdigit():
            digit_map = {"1": "I", "2": "II", "3": "III", "4": "IV"}
            return digit_map.get(core, core)
        val = _roman_to_int(core)
        if val is not None:
            return core

    # Bare roman/numeric stage.
    m = re.fullmatch(r"([ivx]+|[1-4])\s*([abc])?", low, re.I)
    if m:
        core = m.group(1).upper()
        if core.isdigit():
            digit_map = {"1": "I", "2": "II", "3": "III", "4": "IV"}
            return digit_map.get(core, core)
        val = _roman_to_int(core)
        if val is not None:
            return core

    return re.sub(r"\s+", " ", raw.upper()).strip()


def _norm_histology(v: Any) -> Optional[str]:
    """Normalize common breast cancer histology labels."""
    s = _norm_simple(v)
    if s is None:
        return None

    low = s.lower()

    if "dcis" in low or "ductal carcinoma in situ" in low:
        return "DCIS"
    if "invasive ductal" in low or low == "idc":
        return "IDC"
    if "invasive lobular" in low or low == "ilc":
        return "ILC"
    if "mixed" in low:
        return "Mixed"

    return s


def _normalize_for_field(field: str, value: Any) -> Optional[str]:
    """Field-aware normalization before metric calculation."""
    if field in {"er_status", "pr_status", "her2_final_status", "metastatic_disease", "recurrence"}:
        return _norm_status(value)

    if field in {"stage_path", "stage_clinical"}:
        return _norm_stage(value)

    if field == "histology":
        return _norm_histology(value)

    if field in {"er_percent", "pr_percent", "ki67_percent", "grade"}:
        s = _norm_simple(value)
        if s is None:
            return None

        # Pull the first integer if the gold file uses "18%" etc.
        m = re.search(r"\d+", s)
        if m:
            return str(int(m.group(0)))

        return s

    return _norm_simple(value)


def evaluate_patient_labels(
    pred_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    *,
    patient_col: str = "patient_id",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compares patient-level predicted labels against gold labels.

    Gold columns should use: <field>__gold
    Example:
        patient_id, er_status__gold, pr_status__gold, her2_final_status__gold

    Returns:
        metrics_df: precision/recall/F1/support by field
        merged_df: row-level merged predictions/gold values for review, including
                   normalized comparison columns and match flags.
    """
    if patient_col not in pred_df.columns:
        raise ValueError(f"Prediction dataframe missing {patient_col}")
    if patient_col not in gold_df.columns:
        raise ValueError(f"Gold dataframe missing {patient_col}")

    merged = pred_df.merge(gold_df, on=patient_col, how="inner", suffixes=("", "__gold_source"))
    rows: List[dict] = []

    gold_cols = [c for c in gold_df.columns if c.endswith("__gold")]
    for gold_col in gold_cols:
        field = gold_col.replace("__gold", "")
        if field not in merged.columns:
            continue

        pred_norm_col = f"{field}__pred_norm"
        gold_norm_col = f"{field}__gold_norm"
        match_col = f"{field}__match"

        merged[pred_norm_col] = merged[field].apply(lambda x: _normalize_for_field(field, x))
        merged[gold_norm_col] = merged[gold_col].apply(lambda x: _normalize_for_field(field, x))

        tmp = merged[[patient_col, pred_norm_col, gold_norm_col]].copy()
        tmp = tmp.dropna(subset=[gold_norm_col])

        # Only score rows where the gold label is known/reviewed.
        # Missing predictions are kept and scored as a distinct "__MISSING__" class
        # so false negatives are visible in the metrics.
        if tmp.empty:
            merged[match_col] = None
            rows.append({
                "field": field,
                "n": 0,
                "precision_macro": None,
                "recall_macro": None,
                "f1_macro": None,
                "support": 0,
                "accuracy": None,
            })
            continue

        y_true = tmp[gold_norm_col].astype(str)
        y_pred = tmp[pred_norm_col].fillna("__MISSING__").astype(str)

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )

        matches = y_true.reset_index(drop=True).eq(y_pred.reset_index(drop=True))
        accuracy = float(matches.mean()) if len(matches) else 0.0

        merged[match_col] = merged[pred_norm_col].fillna("__MISSING__").astype(str).eq(
            merged[gold_norm_col].fillna("__GOLD_MISSING__").astype(str)
        )

        rows.append({
            "field": field,
            "n": int(tmp.shape[0]),
            "precision_macro": round(float(precision), 4),
            "recall_macro": round(float(recall), 4),
            "f1_macro": round(float(f1), 4),
            "support": int(tmp.shape[0]),
            "accuracy": round(float(accuracy), 4),
        })

    return pd.DataFrame(rows), merged
