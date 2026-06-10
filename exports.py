from __future__ import annotations

from typing import Any, Dict, Iterable, List

import pandas as pd


def make_cohort_export(
    patient_rows: Iterable[Dict[str, Any]],
    evidence_rows: Iterable[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Long, cohort-friendly export.

    This is not full OMOP. It is an OMOP/cohort-ready bridge format:
    patient/person ID + normalized phenotype field/value + evidence provenance.
    """
    patient_df = pd.DataFrame(list(patient_rows))
    evidence_df = pd.DataFrame(list(evidence_rows))

    if patient_df.empty:
        return pd.DataFrame()

    fields = [
        c for c in patient_df.columns
        if not c.startswith("_")
        and not c.endswith("__source_note_id")
        and not c.endswith("__source_note_type")
        and not c.endswith("__source_note_date")
        and c not in {"patient_id"}
        and "__review" not in c
        and not c.endswith("_confidence")
    ]

    rows: List[Dict[str, Any]] = []
    for _, r in patient_df.iterrows():
        patient_id = r.get("patient_id")
        for field in fields:
            value = r.get(field)

            # Skip null/blank values, including pandas/numpy NaN values
            # created by CSV round-trips. Without pd.isna(), blank numeric
            # phenotype fields can leak into the cohort export as rows with
            # normalized_value = NaN.
            if value is None or pd.isna(value) or str(value).strip() == "":
                continue

            field_evidence = pd.DataFrame()
            if not evidence_df.empty and {"patient_id", "field"}.issubset(evidence_df.columns):
                field_evidence = evidence_df[
                    (evidence_df["patient_id"].astype(str) == str(patient_id))
                    & (evidence_df["field"].astype(str) == str(field))
                ]

            if not field_evidence.empty:
                for _, ev in field_evidence.iterrows():
                    rows.append({
                        "person_id": patient_id,
                        "patient_id": patient_id,
                        "phenotype_field": field,
                        "normalized_value": value,
                        "evidence_date": ev.get("note_date"),
                        "source_note_id": ev.get("note_id"),
                        "source_note_type": ev.get("note_type"),
                        "confidence": ev.get("confidence"),
                        "rule_id": ev.get("label"),
                        "engine": ev.get("engine", "medspacy_rules_v1"),
                        "supporting_snippet": ev.get("snippet"),
                        "review_status": r.get(f"{field}__review_status", "unreviewed"),
                        "reviewed_value": r.get(f"{field}__reviewed_value", ""),
                    })
            else:
                rows.append({
                    "person_id": patient_id,
                    "patient_id": patient_id,
                    "phenotype_field": field,
                    "normalized_value": value,
                    "evidence_date": r.get(f"{field}__source_note_date"),
                    "source_note_id": r.get(f"{field}__source_note_id"),
                    "source_note_type": r.get(f"{field}__source_note_type"),
                    "confidence": r.get(f"{field}_confidence"),
                    "rule_id": "",
                    "engine": "aggregation",
                    "supporting_snippet": "",
                    "review_status": r.get(f"{field}__review_status", "unreviewed"),
                    "reviewed_value": r.get(f"{field}__reviewed_value", ""),
                })

    return pd.DataFrame(rows)
