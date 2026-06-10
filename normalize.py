from __future__ import annotations

import re
from typing import Optional, Tuple


# ----------------------------
# Basic parsers
# ----------------------------
def normalize_percent(text: str) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"(\d{1,3})\s*%?", str(text))
    if not m:
        return None
    val = int(m.group(1))
    if 0 <= val <= 100:
        return val
    return None


# ----------------------------
# ER/PR normalization
# ----------------------------
def normalize_status(text: Optional[str]) -> Optional[str]:
    """
    Canonicalize status-like values into:
      Positive | Negative | Equivocal | Unknown | None
    """
    if text is None:
        return None
    t = str(text).strip().lower()
    if t == "":
        return None

    # common positives
    if t in {"positive", "pos", "+", "yes", "detected"}:
        return "Positive"

    # common negatives
    if t in {"negative", "neg", "-", "no", "not detected"}:
        return "Negative"

    # equivocal / borderline
    if t in {"equivocal", "borderline", "indeterminate"}:
        return "Equivocal"

    # unknown / pending
    if t in {"unknown", "pending", "n/a", "na"}:
        return "Unknown"

    # sometimes tokens come like "er+" or "pr-" etc.
    if t.endswith("+"):
        return "Positive"
    if t.endswith("-"):
        return "Negative"

    return None


# ----------------------------
# HER2 helpers
# ----------------------------
def ihc_to_status(ihc: Optional[str]) -> Optional[str]:
    """
    Map IHC score to status.
    """
    if not ihc:
        return None
    s = str(ihc).strip().lower()

    # normalize whitespace like "3 +" -> "3+"
    s = re.sub(r"\s+", "", s)

    # allow "0/1+" and similar
    if s in {"3+", "3"}:
        return "Positive"
    if s in {"2+", "2"}:
        return "Equivocal"
    if s in {"1+", "1", "0", "0+"}:
        return "Negative"
    if s in {"0/1+", "0-1+", "0-1"}:
        return "Negative"

    return None


def fish_to_status(fish: Optional[str]) -> Optional[str]:
    """
    Map FISH text to status.
    """
    if not fish:
        return None
    s = str(fish).strip().lower()

    # strong negatives
    if "not amplified" in s or "nonampl" in s or "non-ampl" in s:
        return "Negative"
    if "no ampl" in s or "no amplification" in s:
        return "Negative"

    # strong positives
    if "amplified" in s or "amplification" in s:
        return "Positive"

    # fallback keywords
    if "positive" in s or "pos" in s:
        return "Positive"
    if "negative" in s or "neg" in s:
        return "Negative"

    return None


def her2_final_status(ihc_score: Optional[str], fish: Optional[str], explicit: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (final_status, confidence_source) where confidence_source is:
      fish | ihc | explicit | None
    """
    fish_status = fish_to_status(fish)
    if fish_status:
        return fish_status, "fish"

    ihc_status = ihc_to_status(ihc_score)
    if ihc_status:
        return ihc_status, "ihc"

    exp = normalize_status(explicit)
    if exp in {"Positive", "Negative", "Equivocal"}:
        return exp, "explicit"

    return None, None


def reconcile_her2(
    ihc_score: Optional[str],
    fish: Optional[str],
    explicit: Optional[str]
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (her2_status, her2_ihc_score, her2_fish) using typical clinical precedence."""
    ihc_score = ihc_score.strip() if ihc_score else None
    fish = fish.strip() if fish else None
    explicit = explicit.strip() if explicit else None

    # FISH overrides IHC
    fish_status = fish_to_status(fish)
    if fish_status:
        return fish_status, ihc_score, fish

    # Next: IHC score mapping
    ihc_status = ihc_to_status(ihc_score)
    if ihc_status:
        return ihc_status, ihc_score, fish

    # Next: explicit text positive/negative/equivocal
    exp = normalize_status(explicit)
    if exp in {"Positive", "Negative", "Equivocal"}:
        return exp, ihc_score, fish

    return None, ihc_score, fish


# ----------------------------
# Histology, grade, stage
# ----------------------------
def normalize_histology(text: str) -> Optional[str]:
    if not text:
        return None
    t = str(text).lower()

    # mixed should beat single labels if both appear
    has_idc = ("invasive ductal" in t) or ("idc" in t)
    has_ilc = ("invasive lobular" in t) or ("ilc" in t)

    if has_idc and has_ilc:
        return "Mixed"
    if "dcis" in t or "ductal carcinoma in situ" in t:
        return "DCIS"
    if has_idc:
        return "IDC"
    if has_ilc:
        return "ILC"

    return None


def normalize_grade(text: str) -> Optional[str]:
    if not text:
        return None
    t = str(text).lower()

    # handles "grade: 3", "grade 2", "histologic grade 1"
    m = re.search(r"\bgrade\s*[:\-]?\s*(\d)\b", t)
    if m:
        g = m.group(1)
        if g in {"1", "2", "3"}:
            return g

    return None


def normalize_stage(text: str) -> Optional[str]:
    if not text:
        return None

    # accept stage i/ii/iii/iv with optional letters
    m = re.search(r"\bstage\s*([ivx]+)\s*([abc])?\b", str(text).lower())
    if not m:
        return None

    roman = m.group(1).upper()
    suffix = (m.group(2) or "").upper()

    # keep as e.g. II, IIIA
    return roman + suffix
