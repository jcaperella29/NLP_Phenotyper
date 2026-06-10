from __future__ import annotations
import os
from typing import List, Dict, Any, Optional
import pandas as pd

def load_mapping_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    return df

def normalize_mapping(df: pd.DataFrame) -> pd.DataFrame:
    # Accept note_id OR filename as identifier
    cols = set(df.columns)
    if "note_id" not in cols and "filename" not in cols:
        raise ValueError("Mapping CSV must include 'note_id' or 'filename' column.")
    if "patient_id" not in cols:
        raise ValueError("Mapping CSV must include 'patient_id' column.")
    for c in ["note_date", "note_type"]:
        if c not in cols:
            df[c] = None
    return df
