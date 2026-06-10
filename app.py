from __future__ import annotations

import base64
import io
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from dash import Dash, html, dcc, Input, Output, State, ALL, ctx, dash_table, no_update

from phenotyper.aggregate import aggregate_patient
from phenotyper.adapters.registry import run_engine
from phenotyper.templates.registry import load_template, list_templates
from phenotyper.validation import add_review_columns, evaluate_patient_labels
from phenotyper.exports import make_cohort_export
from phenotyper.schema import PHENOTYPE_COLUMNS_V1

APP_TITLE = "Breast Cancer Phenotyper (medspaCy/scispaCy MVP)"


NO_PHI_WARNING = (
    "Demo MVP for synthetic or de-identified notes only. "
    "Do not upload real patient records or PHI."
)


def has_extension(filename: str | None, suffix: str) -> bool:
    return bool(filename) and str(filename).lower().endswith(suffix.lower())


def is_spacy_model_available(model_name: str | None) -> tuple[bool, str]:
    """Return whether a selected spaCy/scispaCy model can actually be loaded."""
    model_name = model_name or "en_core_web_sm"
    try:
        import spacy
        spacy.load(model_name)
        return True, f"Model ready: {model_name}"
    except Exception as e:
        return False, f"Model missing or failed to load: {model_name} ({e})"


def order_patient_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Put the demo-critical biomarkers first, then keep all audit/review columns."""
    if df.empty:
        return df

    front = [c for c in PHENOTYPE_COLUMNS_V1 if c in df.columns]
    extra_front = [
        "er_status__review_status",
        "pr_status__review_status",
        "her2_final_status__review_status",
        "ki67_percent__review_status",
        "grade__review_status",
        "stage_path__review_status",
        "stage_clinical__review_status",
        "metastatic_disease__review_status",
    ]
    front.extend([c for c in extra_front if c in df.columns and c not in front])
    rest = [c for c in df.columns if c not in front]
    return df[front + rest]


def make_metric_card(label: str, value: object) -> html.Div:
    return html.Div(
        className="card",
        style={"flex": "1", "minWidth": "150px", "margin": "0", "padding": "10px"},
        children=[
            html.Div(str(value), style={"fontSize": "24px", "fontWeight": "700"}),
            html.Div(label, className="small"),
        ],
    )


def get_available_spacy_models():
    """Return installed spaCy/scispaCy model names for the extraction dropdown."""
    try:
        from spacy.util import get_installed_models

        models = sorted(str(model_name) for model_name in get_installed_models())
    except Exception:
        models = []

    preferred_order = [
        "en_core_web_sm",
        "en_core_web_md",
        "en_core_web_lg",
        "en_core_sci_sm",
        "en_core_sci_md",
        "en_ner_bc5cdr_md",
        "en_ner_bionlp13cg_md",
    ]

    ordered = [m for m in preferred_order if m in models]
    ordered.extend([m for m in models if m not in ordered])

    if not ordered:
        ordered = ["en_core_web_sm"]

    return ordered


def get_available_templates():
    """Return phenotype template options for the template dropdown."""
    try:
        templates = list_templates()
    except Exception:
        templates = []

    if not templates:
        templates = [
            {
                "id": "breast_cancer_biomarkers_v1",
                "label": "Breast Cancer Biomarkers / Status v1",
            }
        ]

    return templates


def file_to_dash_contents(path: Path, mime_type: str) -> str:
    """Encode a local server-side file into the same data URL format dcc.Upload uses."""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def find_sample_data_dir() -> Path:
    """Return the app-local sample_data directory."""
    return Path(__file__).resolve().parent / "sample_data"


def load_sample_demo_payload() -> dict:
    """Load app-local demo notes, mapping CSV, and gold labels CSV.

    Expected folder layout:
      sample_data/
        *.txt                      demo notes
        *mapping*.csv              note-to-patient mapping
        *gold*.csv                 gold labels
    """
    sample_dir = find_sample_data_dir()

    if not sample_dir.exists():
        raise FileNotFoundError(f"Sample data folder not found: {sample_dir}")

    note_paths = sorted(sample_dir.glob("*.txt"))
    if not note_paths:
        raise FileNotFoundError(f"No .txt demo notes found in: {sample_dir}")

    mapping_paths = sorted(
        p for p in sample_dir.glob("*.csv")
        if "mapping" in p.name.lower()
    )
    if not mapping_paths:
        raise FileNotFoundError(f"No mapping CSV found in: {sample_dir}")

    gold_paths = sorted(
        p for p in sample_dir.glob("*.csv")
        if "gold" in p.name.lower()
    )

    notes = [
        {
            "filename": p.name,
            "contents": file_to_dash_contents(p, "text/plain"),
        }
        for p in note_paths
    ]

    mapping = {
        "filename": mapping_paths[0].name,
        "contents": file_to_dash_contents(mapping_paths[0], "text/csv"),
    }

    gold = None
    if gold_paths:
        gold = {
            "filename": gold_paths[0].name,
            "contents": file_to_dash_contents(gold_paths[0], "text/csv"),
        }

    return {
        "notes": notes,
        "mapping": mapping,
        "gold": gold,
        "sample_dir": str(sample_dir),
    }


def parse_txt_upload(contents: str) -> str:
    """Dash upload contents: 'data:...;base64,XXXXX'"""
    if not contents:
        return ""
    _, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)
    try:
        return decoded.decode("utf-8")
    except UnicodeDecodeError:
        return decoded.decode("latin-1")


def df_from_upload_csv(contents: str) -> pd.DataFrame:
    _, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)
    return pd.read_csv(io.BytesIO(decoded))


def normalize_mapping_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    cols = set(df.columns)
    if "note_id" not in cols and "filename" not in cols:
        raise ValueError("Mapping CSV must include 'note_id' or 'filename'.")
    if "patient_id" not in cols:
        raise ValueError("Mapping CSV must include 'patient_id'.")
    for c in ["note_date", "note_type"]:
        if c not in cols:
            df[c] = None
    return df


def fill_patient_fields_from_evidence(patient_df: pd.DataFrame, ev_df: pd.DataFrame) -> pd.DataFrame:
    """Backfill patient-level fields from clean mention-level evidence.

    This protects the workbench from extractor/aggregator naming drift.
    Evidence remains the audit trail; patient table gets rescued values.
    """
    if patient_df.empty or ev_df.empty:
        return patient_df

    out = patient_df.copy()

    rescue_fields = [
        "her2_final_status",
        "ki67_percent",
        "grade",
        "stage_path",
        "stage_clinical",
        "metastatic_disease",
    ]

    for field in rescue_fields:
        if field not in out.columns:
            out[field] = None

    for idx, row in out.iterrows():
        pid = row.get("patient_id")
        ev_pid = ev_df[ev_df["patient_id"].astype(str) == str(pid)].copy()
        if ev_pid.empty or "field" not in ev_pid.columns:
            continue

        # Prefer non-negated, non-uncertain evidence where those flags exist.
        clean = ev_pid.copy()
        if "is_negated" in clean.columns:
            clean = clean[clean["is_negated"].fillna(False) == False]
        if "is_uncertain" in clean.columns:
            clean = clean[clean["is_uncertain"].fillna(False) == False]

        for field in rescue_fields:
            current = out.at[idx, field] if field in out.columns else None
            if pd.notna(current) and str(current).strip() not in ["", "Unknown", "nan", "NaN"]:
                continue

            hits = clean[clean["field"].astype(str) == field]
            if hits.empty:
                continue

            val = hits.iloc[0].get("value")
            if pd.notna(val) and str(val).strip() not in ["", "Unknown", "nan", "NaN"]:
                out.at[idx, field] = val

                # Add provenance if columns exist or can be created.
                for src_col in ["note_id", "note_type", "note_date"]:
                    out_col = f"{field}__source_{src_col}"
                    if out_col not in out.columns:
                        out[out_col] = None
                    out.at[idx, out_col] = hits.iloc[0].get(src_col)

    return out


def make_validation_comparison(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Convert wide merged prediction/gold rows into reviewer-friendly long rows."""
    if merged_df.empty or "patient_id" not in merged_df.columns:
        return pd.DataFrame()

    rows = []
    gold_cols = [c for c in merged_df.columns if c.endswith("__gold")]
    for _, row in merged_df.iterrows():
        patient_id = row.get("patient_id")
        for gold_col in gold_cols:
            field = gold_col.replace("__gold", "")
            if field not in merged_df.columns:
                continue

            pred = row.get(field)
            gold = row.get(gold_col)
            pred_clean = row.get(f"{field}__pred_clean", pred)
            gold_clean = row.get(f"{field}__gold_clean", gold)

            # Skip fields with no gold label entered.
            if pd.isna(gold_clean) or str(gold_clean).strip() == "":
                continue

            match_value = row.get(f"{field}__match")
            if pd.isna(match_value):
                match_value = str(pred_clean).strip() == str(gold_clean).strip()

            rows.append(
                {
                    "patient_id": patient_id,
                    "field": field,
                    "prediction": "" if pd.isna(pred) else pred,
                    "gold": gold,
                    "match": bool(match_value),
                }
            )

    return pd.DataFrame(rows)


app = Dash(
    __name__,
    title=APP_TITLE,
    assets_folder=str(Path(__file__).resolve().parent / "assets"),
)
server = app.server

app.layout = html.Div(
    style={"maxWidth": "1200px", "margin": "18px auto", "padding": "0 10px"},
    children=[
        html.H1(APP_TITLE),
        html.Div(
            NO_PHI_WARNING,
            role="alert",
            style={
                "background": "#fff3cd",
                "border": "1px solid #ffe69c",
                "borderRadius": "10px",
                "padding": "10px 12px",
                "fontWeight": "700",
                "marginBottom": "10px",
            },
        ),
        html.Div(
            "Upload multiple .txt notes and a mapping CSV (note_id/filename → patient_id, note_date, note_type), or load the sample demo data.",
            className="small",
        ),
        html.Div(
            className="card",
            style={"marginTop": "12px"},
            children=[
                html.H3("Guided demo checklist"),
                html.Ol(
                    [
                        html.Li("Load sample demo data or upload synthetic/de-identified notes."),
                        html.Li("Confirm the mapping CSV and model status are ready."),
                        html.Li("Run extraction."),
                        html.Li("Review summary cards and patient phenotypes."),
                        html.Li("Select a patient and inspect supporting evidence."),
                        html.Li("Run validation against demo gold labels."),
                        html.Li("Download the output bundle."),
                    ],
                    style={"marginBottom": "0"},
                ),
            ],
        ),
        html.Div(
            className="card",
            style={"marginTop": "12px"},
            children=[
                html.H3("Sample demo"),
                html.Button(
                    "Load sample demo data",
                    id="btn-load-demo",
                    n_clicks=0,
                    style={"height": "40px"},
                ),
                html.Div(
                    "Loads all .txt notes from the app-local sample_data folder, the CSV with 'mapping' in its name, and the CSV with 'gold' in its name.",
                    className="small",
                    style={"marginTop": "8px"},
                ),
                html.Div(id="demo-status", className="small", style={"marginTop": "8px"}),
            ],
        ),
        html.Div(
            className="row",
            children=[
                html.Div(
                    className="card col",
                    children=[
                        html.H3("1) Upload notes (.txt)"),
                        dcc.Upload(
                            id="upload-notes",
                            children=html.Div(["Drag and drop or ", html.A("select notes")]),
                            multiple=True,
                            style={
                                "width": "100%",
                                "height": "80px",
                                "lineHeight": "80px",
                                "borderWidth": "2px",
                                "borderStyle": "dashed",
                                "borderRadius": "12px",
                                "textAlign": "center",
                            },
                        ),
                        html.Div(
                            "You can add notes in multiple uploads. Remove individual notes with the X button below.",
                            className="small",
                            style={"marginTop": "8px"},
                        ),
                        html.Div(id="notes-status", className="small", style={"marginTop": "8px"}),
                    ],
                ),
                html.Div(
                    className="card col",
                    children=[
                        html.H3("2) Upload mapping CSV"),
                        dcc.Upload(
                            id="upload-mapping",
                            children=html.Div(["Drag and drop or ", html.A("select mapping CSV")]),
                            multiple=False,
                            style={
                                "width": "100%",
                                "height": "80px",
                                "lineHeight": "80px",
                                "borderWidth": "2px",
                                "borderStyle": "dashed",
                                "borderRadius": "12px",
                                "textAlign": "center",
                            },
                        ),
                        html.Div(id="mapping-status", className="small", style={"marginTop": "8px"}),
                    ],
                ),
            ],
        ),
        html.Div(
            className="card",
            style={"marginTop": "12px"},
            children=[
                html.H3("3) Run extraction + patient aggregation"),
                html.Div(
                    className="row",
                    children=[
                        html.Div(
                            className="col",
                            children=[
                                html.Label("Phenotype template"),
                                dcc.Dropdown(
                                    id="template-id",
                                    options=[
                                        {"label": t.get("label", t["id"]), "value": t["id"]}
                                        for t in get_available_templates()
                                    ],
                                    value=get_available_templates()[0]["id"],
                                    clearable=False,
                                    searchable=True,
                                    style={"width": "100%"},
                                ),
                                html.Div(
                                    "Detected phenotype templates from phenotyper/templates.",
                                    className="small",
                                ),
                            ],
                        ),
                        html.Div(
                            className="col",
                            children=[
                                html.Label("Extraction spaCy/scispaCy model"),
                                dcc.Dropdown(
                                    id="model-name",
                                    options=[
                                        {"label": model_name, "value": model_name}
                                        for model_name in get_available_spacy_models()
                                    ],
                                    value=get_available_spacy_models()[0],
                                    clearable=False,
                                    searchable=True,
                                    style={"width": "100%"},
                                ),
                                html.Div(
                                    "Detected installed spaCy/scispaCy models. Install another model and restart the app to make it appear here.",
                                    className="small",
                                ),
                                html.Div(id="model-status", className="small", style={"marginTop": "8px"}),
                            ],
                        ),
                        html.Div(
                            className="col",
                            children=[
                                html.Label(""),
                                html.Button(
                                    "Run extraction",
                                    id="btn-run",
                                    n_clicks=0,
                                    style={"width": "100%", "height": "40px", "marginTop": "22px"},
                                ),
                                dcc.Loading(
                                    html.Div(id="run-status", className="small", style={"marginTop": "8px"}),
                                    type="default",
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        dcc.Store(id="store-note-files", data=[]),
        dcc.Store(id="store-demo-mapping"),
        dcc.Store(id="store-demo-gold"),
        dcc.Store(id="store-patient"),
        dcc.Store(id="store-evidence"),
        dcc.Store(id="store-notesmeta"),
        dcc.Store(id="store-validation-metrics"),
        dcc.Store(id="store-validation-comparison"),
        html.Div(id="summary-cards", style={"marginTop": "12px"}),
        html.Div(
            className="card",
            style={"marginTop": "12px"},
            children=[
                html.H3("Patient phenotypes"),
                dcc.Loading(
                    dash_table.DataTable(
                        id="tbl-patient",
                        page_size=12,
                        filter_action="native",
                        sort_action="native",
                        row_selectable="single",
                        style_table={"overflowX": "auto"},
                        style_cell={"textAlign": "left", "fontSize": "13px", "padding": "6px"},
                        style_header={"fontWeight": "700"},
                    ),
                    type="default",
                ),
                html.Div(className="small", children="Select a patient row to see evidence mentions below."),
            ],
        ),
        html.Div(
            className="card",
            style={"marginTop": "12px"},
            children=[
                html.H3("Evidence (mentions) for selected patient"),
                dcc.Loading(
                    dash_table.DataTable(
                        id="tbl-evidence",
                        page_size=10,
                        filter_action="native",
                        sort_action="native",
                        style_table={"overflowX": "auto"},
                        style_cell={
                            "textAlign": "left",
                            "fontSize": "13px",
                            "padding": "6px",
                            "whiteSpace": "normal",
                            "height": "auto",
                        },
                        style_header={"fontWeight": "700"},
                    ),
                    type="default",
                ),
            ],
        ),
        html.Div(
            className="card",
            style={"marginTop": "12px"},
            children=[
                html.H3("Validation against gold labels"),
                html.Div(
                    className="row",
                    children=[
                        html.Div(
                            className="col",
                            children=[
                                dcc.Upload(
                                    id="upload-gold",
                                    children=html.Div(["Drag and drop or ", html.A("select gold labels CSV")]),
                                    multiple=False,
                                    style={
                                        "width": "100%",
                                        "height": "70px",
                                        "lineHeight": "70px",
                                        "borderWidth": "2px",
                                        "borderStyle": "dashed",
                                        "borderRadius": "12px",
                                        "textAlign": "center",
                                    },
                                ),
                                html.Div(id="gold-status", className="small", style={"marginTop": "8px"}),
                                html.Div(
                                    "Gold columns should use field__gold, e.g. er_status__gold, her2_final_status__gold.",
                                    className="small",
                                ),
                            ],
                        ),
                        html.Div(
                            className="col",
                            children=[
                                html.Button(
                                    "Run validation",
                                    id="btn-validate",
                                    n_clicks=0,
                                    style={"width": "100%", "height": "40px", "marginTop": "15px"},
                                ),
                                dcc.Loading(
                                    html.Div(id="validation-status", className="small", style={"marginTop": "8px"}),
                                    type="default",
                                ),
                            ],
                        ),
                    ],
                ),
                html.H4("Field-level metrics"),
                dcc.Loading(
                    dash_table.DataTable(
                        id="tbl-validation-metrics",
                        page_size=12,
                        filter_action="native",
                        sort_action="native",
                        style_table={"overflowX": "auto"},
                        style_cell={"textAlign": "left", "fontSize": "13px", "padding": "6px"},
                        style_header={"fontWeight": "700"},
                    ),
                    type="default",
                ),
                html.H4("Prediction vs gold review"),
                dcc.Loading(
                    dash_table.DataTable(
                        id="tbl-validation-comparison",
                        page_size=12,
                        filter_action="native",
                        sort_action="native",
                        style_table={"overflowX": "auto"},
                        style_cell={"textAlign": "left", "fontSize": "13px", "padding": "6px"},
                        style_header={"fontWeight": "700"},
                    ),
                    type="default",
                ),
            ],
        ),
        html.Div(
            className="card",
            style={"marginTop": "12px"},
            children=[
                html.H3("Export"),
                html.Div(
                    className="row",
                    children=[
                        html.Div(
                            className="col",
                            children=[
                                html.Button("Download patient table CSV", id="btn-dl-patient"),
                                dcc.Download(id="dl-patient"),
                            ],
                        ),
                        html.Div(
                            className="col",
                            children=[
                                html.Button("Download evidence CSV", id="btn-dl-evidence"),
                                dcc.Download(id="dl-evidence"),
                            ],
                        ),
                        html.Div(
                            className="col",
                            children=[
                                html.Button("Download cohort-friendly CSV", id="btn-dl-cohort"),
                                dcc.Download(id="dl-cohort"),
                            ],
                        ),
                        html.Div(
                            className="col",
                            children=[
                                html.Button("Download validation metrics CSV", id="btn-dl-validation-metrics"),
                                dcc.Download(id="dl-validation-metrics"),
                            ],
                        ),
                        html.Div(
                            className="col",
                            children=[
                                html.Button("Download prediction-vs-gold CSV", id="btn-dl-validation-comparison"),
                                dcc.Download(id="dl-validation-comparison"),
                            ],
                        ),
                        html.Div(
                            className="col",
                            children=[
                                html.Button("Download all outputs ZIP", id="btn-dl-bundle"),
                                dcc.Download(id="dl-bundle"),
                            ],
                        ),
                    ],
                ),
                html.Div(id="export-status", className="small", style={"marginTop": "8px"}),
            ],
        ),
    ],
)



@app.callback(
    Output("model-status", "children"),
    Input("model-name", "value"),
)
def render_model_status(model_name):
    ok, message = is_spacy_model_available(model_name)
    return html.Span(
        message,
        style={"color": "#146c43" if ok else "#b02a37", "fontWeight": "700"},
    )


@app.callback(
    Output("summary-cards", "children"),
    Input("store-patient", "data"),
    Input("store-evidence", "data"),
    Input("store-notesmeta", "data"),
    Input("store-validation-metrics", "data"),
)
def render_summary_cards(patient_rows, evidence_rows, notes_meta, validation_metrics_rows):
    if not patient_rows:
        return html.Div(
            className="card",
            children=[
                html.H3("Run summary"),
                html.Div("No extraction run yet. Load demo data or upload synthetic/de-identified files, then click Run extraction.", className="small"),
            ],
        )

    patient_df = pd.DataFrame(patient_rows)
    core_fields = [
        "er_status",
        "pr_status",
        "her2_final_status",
        "ki67_percent",
        "grade",
        "stage_path",
        "stage_clinical",
        "metastatic_disease",
    ]
    populated = 0
    for field in core_fields:
        if field in patient_df.columns:
            populated += int(patient_df[field].notna().sum())

    validation_fields = len(validation_metrics_rows or [])
    cards = [
        make_metric_card("Notes processed", len(notes_meta or [])),
        make_metric_card("Patients", len(patient_rows or [])),
        make_metric_card("Evidence mentions", len(evidence_rows or [])),
        make_metric_card("Core fields populated", populated),
        make_metric_card("Validation fields evaluated", validation_fields),
    ]
    return html.Div(className="row", style={"gap": "8px"}, children=cards)


@app.callback(
    Output("store-note-files", "data"),
    Output("notes-status", "children"),
    Input("upload-notes", "contents"),
    Input("upload-notes", "filename"),
    Input("btn-load-demo", "n_clicks"),
    Input({"type": "remove-note", "index": ALL}, "n_clicks"),
    State("store-note-files", "data"),
)
def manage_uploaded_notes(upload_contents, upload_filenames, demo_clicks, remove_clicks, stored_notes):
    """Keep uploaded notes across multiple drag/drop events and allow per-note removal."""
    notes = list(stored_notes or [])
    triggered = ctx.triggered_id

    if triggered == "btn-load-demo":
        try:
            payload = load_sample_demo_payload()
            notes = payload["notes"]
        except Exception as e:
            return notes, f"Demo load error: {e}"

    elif isinstance(triggered, dict) and triggered.get("type") == "remove-note":
        remove_filename = triggered.get("index")
        notes = [n for n in notes if n.get("filename") != remove_filename]

    elif triggered in {"upload-notes", None} and upload_contents and upload_filenames:
        # Normalize Dash single-file vs multi-file payloads.
        if isinstance(upload_filenames, str):
            upload_filenames = [upload_filenames]
        if isinstance(upload_contents, str):
            upload_contents = [upload_contents]

        existing_by_name = {n.get("filename"): n for n in notes}

        rejected = []
        for filename, contents in zip(upload_filenames, upload_contents):
            if not filename:
                continue
            if not has_extension(filename, ".txt"):
                rejected.append(filename)
                continue

            # Re-uploading the same filename replaces the stored contents instead
            # of creating duplicate rows.
            existing_by_name[filename] = {
                "filename": filename,
                "contents": contents,
            }

        # Preserve previous display order, then append truly new files.
        old_order = [n.get("filename") for n in notes if n.get("filename") in existing_by_name]
        new_names = [fn for fn in upload_filenames if fn not in old_order]
        final_order = old_order + new_names
        notes = [existing_by_name[fn] for fn in final_order if fn in existing_by_name]

    if not notes:
        return [], "No notes uploaded yet."

    note_items = []
    for n in notes:
        fn = n.get("filename", "")
        note_items.append(
            html.Div(
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "space-between",
                    "gap": "8px",
                    "padding": "4px 0",
                    "borderBottom": "1px solid #eee",
                },
                children=[
                    html.Span(fn),
                    html.Button(
                        "X",
                        id={"type": "remove-note", "index": fn},
                        n_clicks=0,
                        title=f"Remove {fn}",
                        style={
                            "border": "1px solid #ccc",
                            "borderRadius": "6px",
                            "background": "white",
                            "cursor": "pointer",
                            "fontWeight": "700",
                        },
                    ),
                ],
            )
        )

    status_children = [html.Div(f"{len(notes)} note(s) ready."), html.Div(note_items, style={"marginTop": "6px"})]
    if "rejected" in locals() and rejected:
        status_children.insert(1, html.Div("Ignored non-.txt file(s): " + ", ".join(rejected), style={"color": "#b02a37"}))

    return notes, html.Div(children=status_children)


@app.callback(
    Output("gold-status", "children"),
    Input("upload-gold", "filename"),
    Input("store-demo-gold", "data"),
)
def show_gold_status(filename, demo_gold):
    if filename:
        if not has_extension(filename, ".csv"):
            return f"Gold labels must be a .csv file. Selected: {filename}"
        return f"Gold labels ready: {filename}"
    if demo_gold and demo_gold.get("filename"):
        return f"Demo gold labels loaded: {demo_gold.get('filename')}"
    return "No gold labels uploaded yet."


@app.callback(
    Output("mapping-status", "children"),
    Input("upload-mapping", "filename"),
    Input("store-demo-mapping", "data"),
)
def show_mapping_status(filename, demo_mapping):
    if filename:
        if not has_extension(filename, ".csv"):
            return f"Mapping file must be a .csv file. Selected: {filename}"
        return f"Mapping file ready: {filename}"
    if demo_mapping and demo_mapping.get("filename"):
        return f"Demo mapping loaded: {demo_mapping.get('filename')}"
    return "No mapping uploaded yet."


@app.callback(
    Output("store-demo-mapping", "data"),
    Output("store-demo-gold", "data"),
    Output("demo-status", "children"),
    Input("btn-load-demo", "n_clicks"),
    prevent_initial_call=True,
)
def load_demo_supporting_files(n_clicks):
    try:
        payload = load_sample_demo_payload()
        gold = payload.get("gold")
        gold_msg = gold.get("filename") if gold else "no gold CSV found"
        return (
            payload["mapping"],
            gold,
            f"Loaded sample demo from {payload['sample_dir']} | Notes: {len(payload['notes'])} | Mapping: {payload['mapping']['filename']} | Gold: {gold_msg}",
        )
    except Exception as e:
        return no_update, no_update, f"Demo load error: {e}"


@app.callback(
    Output("store-patient", "data"),
    Output("store-evidence", "data"),
    Output("store-notesmeta", "data"),
    Output("run-status", "children"),
    Input("btn-run", "n_clicks"),
    State("store-note-files", "data"),
    State("upload-mapping", "contents"),
    State("upload-mapping", "filename"),
    State("store-demo-mapping", "data"),
    State("template-id", "value"),
    State("model-name", "value"),
    prevent_initial_call=True,
)
def run_pipeline(n_clicks, stored_notes, mapping_contents, mapping_filename, demo_mapping, template_id, model_name):
    if not stored_notes:
        return no_update, no_update, no_update, "Please upload at least one .txt note."

    note_contents = [n.get("contents") for n in stored_notes if n.get("contents")]
    note_filenames = [n.get("filename") for n in stored_notes if n.get("filename")]

    if not note_contents or not note_filenames:
        return no_update, no_update, no_update, "Please upload at least one .txt note."

    invalid_notes = [fn for fn in note_filenames if not has_extension(fn, ".txt")]
    if invalid_notes:
        return no_update, no_update, no_update, "Only .txt notes are supported. Remove: " + ", ".join(invalid_notes[:5])

    model_ok, model_msg = is_spacy_model_available(model_name or "en_core_web_sm")
    if not model_ok:
        return no_update, no_update, no_update, model_msg

    # Load mapping. Manual upload wins; demo mapping is used as fallback.
    if not mapping_contents and demo_mapping:
        mapping_contents = demo_mapping.get("contents")
        mapping_filename = demo_mapping.get("filename")

    mapping_df = None
    if mapping_contents:
        if mapping_filename and not has_extension(mapping_filename, ".csv"):
            return no_update, no_update, no_update, f"Mapping file must be .csv. Selected: {mapping_filename}"
        try:
            mapping_df = normalize_mapping_df(df_from_upload_csv(mapping_contents))
        except Exception as e:
            return no_update, no_update, no_update, f"Mapping CSV error: {e}"
    else:
        return no_update, no_update, no_update, "Please upload a mapping CSV or click Load sample demo data."

    # Build per-note metadata table
    rows_meta = []
    unmatched_uploaded_notes = []

    uploaded_note_ids = {os.path.splitext(fn)[0] for fn in note_filenames}
    uploaded_filenames = set(note_filenames)

    for fn in note_filenames:
        note_id = os.path.splitext(fn)[0]
        patient_id = note_id
        note_date = None
        note_type = "Unknown"

        matched_mapping = False

        if mapping_df is not None:
            if "note_id" in mapping_df.columns:
                hit = mapping_df[mapping_df["note_id"].astype(str) == str(note_id)]
            else:
                hit = mapping_df[mapping_df["filename"].astype(str) == str(fn)]
            if len(hit) >= 1:
                matched_mapping = True
                h = hit.iloc[0]
                patient_id = str(h["patient_id"])
                note_date = None if pd.isna(h.get("note_date")) else str(h.get("note_date"))
                note_type = "Unknown" if pd.isna(h.get("note_type")) else str(h.get("note_type"))

        if mapping_df is not None and not matched_mapping:
            unmatched_uploaded_notes.append(fn)

        rows_meta.append(
            {
                "filename": fn,
                "note_id": note_id,
                "patient_id": patient_id,
                "note_date": note_date,
                "note_type": note_type,
            }
        )

    meta_df = pd.DataFrame(rows_meta)

    # Detect mapping rows that do not correspond to currently uploaded notes.
    # This is only a warning, not a hard failure, because users may intentionally
    # upload a subset of the sample/demo notes.
    unused_mapping_rows = []
    if mapping_df is not None:
        if "note_id" in mapping_df.columns:
            mapped_ids = set(mapping_df["note_id"].dropna().astype(str))
            unused_mapping_rows = sorted(mapped_ids - uploaded_note_ids)
        elif "filename" in mapping_df.columns:
            mapped_files = set(mapping_df["filename"].dropna().astype(str))
            unused_mapping_rows = sorted(mapped_files - uploaded_filenames)

    # Extract note-level phenotypes + evidence
    template_id = template_id or "breast_cancer_biomarkers_v1"
    try:
        template = load_template(template_id)
    except Exception as e:
        return no_update, no_update, no_update, f"Template error: {e}"
    engine_name = "medspacy_rules_v1"

    note_pheno_rows = []
    evidence_objs = []  # true Evidence objects for aggregation scoring
    ev_rows = []        # dicts for UI table + downloads

    failed_notes = []
    for contents, fn in zip(note_contents, note_filenames):
        try:
            note_text = parse_txt_upload(contents)
            meta = meta_df[meta_df["filename"] == fn].iloc[0].to_dict()

            phenos, evidence = run_engine(
                engine_name,
                note_text,
                patient_id=meta["patient_id"],
                note_id=meta["note_id"],
                note_date=meta.get("note_date"),
                note_type=meta.get("note_type"),
                model_name=model_name or "en_core_web_sm",
                template=template,
            )

            note_pheno_rows.append(phenos)
            evidence_objs.extend(evidence)
            ev_rows.extend([e.to_dict() for e in evidence])
        except Exception as e:
            failed_notes.append({"filename": fn, "error": str(e)})
            continue

    if not note_pheno_rows:
        detail = "; ".join(f"{x['filename']}: {x['error']}" for x in failed_notes[:3])
        return no_update, no_update, no_update, "Extraction failed for every note. " + detail

    note_pheno_df = pd.DataFrame(note_pheno_rows)
    ev_df = pd.DataFrame(ev_rows) if ev_rows else pd.DataFrame()

    # Aggregate by patient using true Evidence objects.
    patient_rows = []
    for pid, grp in note_pheno_df.groupby("patient_id", dropna=False):
        patient_rows.append(aggregate_patient(grp.to_dict("records"), evidence_objs))

    patient_df = pd.DataFrame(patient_rows)
    patient_df = fill_patient_fields_from_evidence(patient_df, ev_df)
    patient_df = add_review_columns(patient_df)
    patient_df = order_patient_columns(patient_df)

    status_parts = [
        f"Done. Notes processed: {len(note_filenames)}",
        f"Patients: {patient_df.shape[0]}",
        f"Mentions: {len(ev_rows)}",
        f"Template: {template_id}",
        f"Model: {model_name or 'en_core_web_sm'}",
    ]

    if unmatched_uploaded_notes:
        status_parts.append(
            "Uploaded note(s) without mapping: " + ", ".join(unmatched_uploaded_notes[:5])
            + (" ..." if len(unmatched_uploaded_notes) > 5 else "")
        )

    if unused_mapping_rows:
        status_parts.append(
            "Mapping row(s) not uploaded: " + ", ".join(unused_mapping_rows[:5])
            + (" ..." if len(unused_mapping_rows) > 5 else "")
        )

    if failed_notes:
        shown = "; ".join(f"{x['filename']}: {x['error']}" for x in failed_notes[:3])
        status_parts.append(
            f"Notes failed: {len(failed_notes)} ({shown}{' ...' if len(failed_notes) > 3 else ''})"
        )

    return (
        patient_df.to_dict("records"),
        ev_rows,
        meta_df.to_dict("records"),
        " | ".join(status_parts),
    )


@app.callback(
    Output("tbl-patient", "data"),
    Output("tbl-patient", "columns"),
    Input("store-patient", "data"),
)
def render_patient_table(rows):
    if not rows:
        return [], []
    df = order_patient_columns(pd.DataFrame(rows))
    cols = [{"name": c, "id": c} for c in df.columns]
    return df.to_dict("records"), cols


@app.callback(
    Output("tbl-evidence", "data"),
    Output("tbl-evidence", "columns"),
    Input("tbl-patient", "selected_rows"),
    State("tbl-patient", "data"),
    State("store-evidence", "data"),
)
def render_evidence(selected_rows, patient_rows, ev_rows):
    if not ev_rows:
        return [], []
    ev_df = pd.DataFrame(ev_rows)

    if not selected_rows or not patient_rows:
        cols = [{"name": c, "id": c} for c in ev_df.columns]
        return ev_df.to_dict("records"), cols

    sel_idx = selected_rows[0]
    pid = patient_rows[sel_idx].get("patient_id")
    ev_df = ev_df[ev_df["patient_id"].astype(str) == str(pid)].copy()

    preferred = [
        "patient_id",
        "note_id",
        "note_date",
        "note_type",
        "field",
        "value",
        "label",
        "confidence",
        "is_negated",
        "is_uncertain",
        "snippet",
    ]
    cols = [c for c in preferred if c in ev_df.columns] + [c for c in ev_df.columns if c not in preferred]
    ev_df = ev_df[cols]
    return ev_df.to_dict("records"), [{"name": c, "id": c} for c in ev_df.columns]


@app.callback(
    Output("store-validation-metrics", "data"),
    Output("store-validation-comparison", "data"),
    Output("validation-status", "children"),
    Input("btn-validate", "n_clicks"),
    State("store-patient", "data"),
    State("upload-gold", "contents"),
    State("upload-gold", "filename"),
    State("store-demo-gold", "data"),
    prevent_initial_call=True,
)
def run_validation(n_clicks, patient_rows, gold_contents, gold_filename, demo_gold):
    if not patient_rows:
        return no_update, no_update, "Run extraction before validation."

    # Manual upload wins; demo gold is used as fallback.
    if not gold_contents and demo_gold:
        gold_contents = demo_gold.get("contents")
        gold_filename = demo_gold.get("filename")

    if not gold_contents:
        return no_update, no_update, "Upload a gold-label CSV first or click Load sample demo data."

    if gold_filename and not has_extension(gold_filename, ".csv"):
        return no_update, no_update, f"Gold labels file must be .csv. Selected: {gold_filename}"

    try:
        pred_df = pd.DataFrame(patient_rows)
        gold_df = df_from_upload_csv(gold_contents)
        gold_df.columns = [c.strip() for c in gold_df.columns]

        metrics_df, merged_df = evaluate_patient_labels(pred_df, gold_df)
        comparison_df = make_validation_comparison(merged_df)

        return (
            metrics_df.to_dict("records"),
            comparison_df.to_dict("records"),
            f"Validation complete. Gold patients matched: {merged_df.shape[0]} | Fields evaluated: {metrics_df.shape[0]}",
        )
    except Exception as e:
        return no_update, no_update, f"Validation error: {e}"


@app.callback(
    Output("tbl-validation-metrics", "data"),
    Output("tbl-validation-metrics", "columns"),
    Input("store-validation-metrics", "data"),
)
def render_validation_metrics(rows):
    if not rows:
        return [], []
    df = pd.DataFrame(rows)
    return df.to_dict("records"), [{"name": c, "id": c} for c in df.columns]


@app.callback(
    Output("tbl-validation-comparison", "data"),
    Output("tbl-validation-comparison", "columns"),
    Input("store-validation-comparison", "data"),
)
def render_validation_comparison(rows):
    if not rows:
        return [], []
    df = pd.DataFrame(rows)
    preferred = ["patient_id", "field", "prediction", "gold", "match"]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]
    return df.to_dict("records"), [{"name": c, "id": c} for c in df.columns]


@app.callback(
    Output("dl-patient", "data"),
    Output("export-status", "children", allow_duplicate=True),
    Input("btn-dl-patient", "n_clicks"),
    State("store-patient", "data"),
    prevent_initial_call=True,
)
def download_patient(n, rows):
    if not rows:
        return no_update, "Run extraction before downloading the patient table."
    df = order_patient_columns(pd.DataFrame(rows))
    return dcc.send_data_frame(df.to_csv, "patient_phenotypes_v1.csv", index=False), "Patient table download ready."


@app.callback(
    Output("dl-evidence", "data"),
    Output("export-status", "children", allow_duplicate=True),
    Input("btn-dl-evidence", "n_clicks"),
    State("store-evidence", "data"),
    prevent_initial_call=True,
)
def download_evidence(n, rows):
    if not rows:
        return no_update, "Run extraction before downloading evidence."
    df = pd.DataFrame(rows)
    return dcc.send_data_frame(df.to_csv, "extraction_evidence.csv", index=False), "Evidence download ready."


@app.callback(
    Output("dl-cohort", "data"),
    Output("export-status", "children", allow_duplicate=True),
    Input("btn-dl-cohort", "n_clicks"),
    State("store-patient", "data"),
    State("store-evidence", "data"),
    prevent_initial_call=True,
)
def download_cohort(n, patient_rows, evidence_rows):
    if not patient_rows:
        return no_update, "Run extraction before downloading the cohort-friendly export."
    df = make_cohort_export(patient_rows, evidence_rows or [])
    return dcc.send_data_frame(df.to_csv, "cohort_friendly_export.csv", index=False), "Cohort-friendly export download ready."


@app.callback(
    Output("dl-validation-metrics", "data"),
    Output("export-status", "children", allow_duplicate=True),
    Input("btn-dl-validation-metrics", "n_clicks"),
    State("store-validation-metrics", "data"),
    prevent_initial_call=True,
)
def download_validation_metrics(n, rows):
    if not rows:
        return no_update, "Run validation before downloading validation metrics."
    df = pd.DataFrame(rows)
    return dcc.send_data_frame(df.to_csv, "validation_metrics.csv", index=False), "Validation metrics download ready."


@app.callback(
    Output("dl-validation-comparison", "data"),
    Output("export-status", "children", allow_duplicate=True),
    Input("btn-dl-validation-comparison", "n_clicks"),
    State("store-validation-comparison", "data"),
    prevent_initial_call=True,
)
def download_validation_comparison(n, rows):
    if not rows:
        return no_update, "Run validation before downloading prediction-vs-gold review."
    df = pd.DataFrame(rows)
    return dcc.send_data_frame(df.to_csv, "prediction_vs_gold.csv", index=False), "Prediction-vs-gold download ready."


@app.callback(
    Output("dl-bundle", "data"),
    Output("export-status", "children", allow_duplicate=True),
    Input("btn-dl-bundle", "n_clicks"),
    State("store-patient", "data"),
    State("store-evidence", "data"),
    State("store-validation-metrics", "data"),
    State("store-validation-comparison", "data"),
    prevent_initial_call=True,
)
def download_bundle(n, patient_rows, evidence_rows, validation_metrics_rows, validation_comparison_rows):
    """Download all currently available output tables as one ZIP file."""
    if not patient_rows:
        return no_update, "Run extraction before downloading the output bundle."

    patient_df = pd.DataFrame(patient_rows)
    evidence_df = pd.DataFrame(evidence_rows or [])
    cohort_df = make_cohort_export(patient_rows, evidence_rows or [])

    validation_metrics_df = pd.DataFrame(validation_metrics_rows or [])
    validation_comparison_df = pd.DataFrame(validation_comparison_rows or [])

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("patient_phenotypes_v1.csv", patient_df.to_csv(index=False))
        zf.writestr("extraction_evidence.csv", evidence_df.to_csv(index=False))
        zf.writestr("cohort_friendly_export.csv", cohort_df.to_csv(index=False))

        if validation_metrics_rows:
            zf.writestr("validation_metrics.csv", validation_metrics_df.to_csv(index=False))

        if validation_comparison_rows:
            zf.writestr("prediction_vs_gold.csv", validation_comparison_df.to_csv(index=False))

        manifest_lines = [
            "Breast Cancer Phenotyper output bundle",
            "",
            f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
            "App version: demo-hardening-v1",
            "",
            "Included files:",
            "- patient_phenotypes_v1.csv",
            "- extraction_evidence.csv",
            "- cohort_friendly_export.csv",
        ]

        if validation_metrics_rows:
            manifest_lines.append("- validation_metrics.csv")
        else:
            manifest_lines.append("- validation_metrics.csv not included because validation has not been run.")

        if validation_comparison_rows:
            manifest_lines.append("- prediction_vs_gold.csv")
        else:
            manifest_lines.append("- prediction_vs_gold.csv not included because validation has not been run.")

        manifest_lines.extend(
            [
                "",
                f"Patients: {len(patient_rows)}",
                f"Evidence rows: {len(evidence_rows or [])}",
                f"Cohort rows: {cohort_df.shape[0]}",
            ]
        )

        zf.writestr("README_bundle.txt", "\n".join(manifest_lines))

    buffer.seek(0)
    return dcc.send_bytes(buffer.getvalue(), "phenotyper_outputs_bundle.zip"), "Output bundle download ready."


if __name__ == "__main__":
    debug = os.getenv("DASH_DEBUG", "false").strip().lower() in {"1", "true", "yes"}
    port = int(os.getenv("PORT", "8050"))
    app.run(debug=debug, host="0.0.0.0", port=port)

