from __future__ import annotations

# ----------------------------
# Phenotype schema (v1)
# ----------------------------
# v1 goals:
# - Stable column names
# - Auditability (sources + confidence)
# - Deterministic, rule-first (no ML required for core biomarkers)
# ----------------------------

PHENOTYPE_COLUMNS_V1 = [
    # identity
    "patient_id",

    # ER / PR
    "er_status",
    "er_percent",
    "er_confidence",

    "pr_status",
    "pr_percent",
    "pr_confidence",

    # HER2
    "her2_ihc_score",
    "her2_fish",
    "her2_final_status",
    "her2_confidence",

    # proliferation
    "ki67_percent",
    "ki67_confidence",

    # tumor biology
    "histology",
    "histology_confidence",
    "grade",

    # stage
    "stage_clinical",
    "stage_path",
]

# Backwards compatibility (if other code imports PHENOTYPE_COLUMNS)
PHENOTYPE_COLUMNS = PHENOTYPE_COLUMNS_V1


# ----------------------------
# Enumerations / canonical values
# ----------------------------
ENUM_STATUS = {"Positive", "Negative", "Equivocal", "Unknown", None}

ENUM_CONFIDENCE_SOURCE = {
    "pathology",
    "addendum",
    "consult",
    "radiology",
    "unknown",
    None,
}


# ----------------------------
# Note type precedence
# Higher is better.
#
# IMPORTANT: keep these keys consistent with whatever your note ingestion uses
# (e.g., "Pathology", "SurgicalPathology", "Addendum", "OncologyConsult", etc.)
# ----------------------------
NOTE_TYPE_PRECEDENCE = {
    # pathology & addenda should dominate tumor biology fields
    "Pathology": 100,
    "SurgicalPathology": 100,
    "PathologyAddendum": 100,
    "Addendum": 100,

    # clinician summary (usually reflects pathology, but can be earlier/hedgier)
    "OncologyConsult": 70,

    # imaging is lowest confidence for biomarkers
    "Radiology": 40,

    # general notes
    "ProgressNote": 30,

    # unknowns
    "Unknown": 0,
    "": 0,
    None: 0,
}
