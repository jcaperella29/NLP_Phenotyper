from __future__ import annotations

import base64
import io
import os
import pandas as pd
from dash import Dash, html, dcc, Input, Output, State, dash_table, no_update

from phenotyper.extract import extract_note
from phenotyper.aggregate import aggregate_patient

APP_TITLE = "Breast Cancer Phenotyper (medspaCy/scispaCy MVP)"


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


app = Dash(__name__, title=APP_TITLE)
server = app.server

app.layout = html.Div(
    style={"maxWidth": "1200px", "margin": "18px auto", "padding": "0 10px"},
    children=[
        html.H1(APP_TITLE),
        html.Div(
            "Upload multiple .txt notes. Optionally upload a mapping CSV (note_id/filename → patient_id, note_date, note_type).",
            className="small",
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
                        html.Div(id="notes-status", className="small", style={"marginTop": "8px"}),
                    ],
                ),
                html.Div(
                    className="card col",
                    children=[
                        html.H3("2) Upload mapping CSV (optional)"),
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
                                html.Label("spaCy base model name"),
                                dcc.Input(
                                    id="model-name",
                                    value="en_core_web_sm",
                                    type="text",
                                    style={"width": "100%"},
                                ),
                                html.Div(
                                    "Tip: start with en_core_web_sm. You can later swap to a scispaCy model.",
                                    className="small",
                                ),
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
                                html.Div(id="run-status", className="small", style={"marginTop": "8px"}),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        dcc.Store(id="store-patient"),
        dcc.Store(id="store-evidence"),
        dcc.Store(id="store-notesmeta"),
        html.Div(
            className="card",
            style={"marginTop": "12px"},
            children=[
                html.H3("Patient phenotypes"),
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
                html.Div(className="small", children="Select a patient row to see evidence mentions below."),
            ],
        ),
        html.Div(
            className="card",
            style={"marginTop": "12px"},
            children=[
                html.H3("Evidence (mentions) for selected patient"),
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
                    ],
                ),
            ],
        ),
    ],
)


@app.callback(
    Output("notes-status", "children"),
    Input("upload-notes", "filename"),
)
def show_notes_status(filenames):
    if not filenames:
        return "No notes uploaded yet."
    return f"{len(filenames)} note(s) ready: " + ", ".join(filenames[:6]) + (" ..." if len(filenames) > 6 else "")


@app.callback(
    Output("mapping-status", "children"),
    Input("upload-mapping", "filename"),
)
def show_mapping_status(filename):
    if not filename:
        return "No mapping uploaded (optional)."
    return f"Mapping file ready: {filename}"


@app.callback(
    Output("store-patient", "data"),
    Output("store-evidence", "data"),
    Output("store-notesmeta", "data"),
    Output("run-status", "children"),
    Input("btn-run", "n_clicks"),
    State("upload-notes", "contents"),
    State("upload-notes", "filename"),
    State("upload-mapping", "contents"),
    State("upload-mapping", "filename"),
    State("model-name", "value"),
    prevent_initial_call=True,
)
def run_pipeline(n_clicks, note_contents, note_filenames, mapping_contents, mapping_filename, model_name):
    if not note_contents or not note_filenames:
        return no_update, no_update, no_update, "Please upload at least one .txt note."

    # Load mapping if present
    mapping_df = None
    if mapping_contents:
        try:
            mapping_df = normalize_mapping_df(df_from_upload_csv(mapping_contents))
        except Exception as e:
            return no_update, no_update, no_update, f"Mapping CSV error: {e}"

    # Build per-note metadata table
    rows_meta = []
    for fn in note_filenames:
        note_id = os.path.splitext(fn)[0]
        patient_id = note_id
        note_date = None
        note_type = "Unknown"

        if mapping_df is not None:
            if "note_id" in mapping_df.columns:
                hit = mapping_df[mapping_df["note_id"].astype(str) == str(note_id)]
            else:
                hit = mapping_df[mapping_df["filename"].astype(str) == str(fn)]
            if len(hit) >= 1:
                h = hit.iloc[0]
                patient_id = str(h["patient_id"])
                note_date = None if pd.isna(h.get("note_date")) else str(h.get("note_date"))
                note_type = "Unknown" if pd.isna(h.get("note_type")) else str(h.get("note_type"))

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

    # Extract note-level phenotypes + evidence
    note_pheno_rows = []
    ev_rows = []  # dicts for UI table + downloads

    for contents, fn in zip(note_contents, note_filenames):
        note_text = parse_txt_upload(contents)
        meta = meta_df[meta_df["filename"] == fn].iloc[0].to_dict()

        phenos, evidence = extract_note(
            note_text,
            patient_id=meta["patient_id"],
            note_id=meta["note_id"],
            note_date=meta.get("note_date"),
            note_type=meta.get("note_type"),
            model_name=model_name or "en_core_web_sm",
        )

        note_pheno_rows.append(phenos)
        ev_rows.extend([e.to_dict() for e in evidence])

    note_pheno_df = pd.DataFrame(note_pheno_rows)
    ev_df = pd.DataFrame(ev_rows) if ev_rows else pd.DataFrame()

    # Rebuild Evidence objects for aggregation scoring (so aggregate_patient can use evidence flags)
    evidence_objs = []
    if not ev_df.empty:
        # Evidence dataclass in your project expects:
        # patient_id, note_id, note_date, note_type, field, value, start, end, snippet, label, confidence, is_negated, is_uncertain
        # We'll pass only what exists; missing keys default safely.
        for r in ev_df.to_dict("records"):
            try:
                from phenotyper.evidence import Evidence  # local import avoids circulars
                evidence_objs.append(Evidence(**r))
            except Exception:
                # If evidence schema differs, aggregation will still work (it just won't evidence-score).
                pass

    # Aggregate by patient (IMPORTANT: pass evidence_objs, not [])
    patient_rows = []
    for pid, grp in note_pheno_df.groupby("patient_id", dropna=False):
        patient_rows.append(aggregate_patient(grp.to_dict("records"), evidence_objs))

    patient_df = pd.DataFrame(patient_rows)

    return (
        patient_df.to_dict("records"),
        ev_rows,
        meta_df.to_dict("records"),
        f"Done. Notes processed: {len(note_filenames)} | Patients: {patient_df.shape[0]} | Mentions: {len(ev_rows)}",
    )


@app.callback(
    Output("tbl-patient", "data"),
    Output("tbl-patient", "columns"),
    Input("store-patient", "data"),
)
def render_patient_table(rows):
    if not rows:
        return [], []
    df = pd.DataFrame(rows)
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
    Output("dl-patient", "data"),
    Input("btn-dl-patient", "n_clicks"),
    State("store-patient", "data"),
    prevent_initial_call=True,
)
def download_patient(n, rows):
    if not rows:
        return no_update
    df = pd.DataFrame(rows)
    return dcc.send_data_frame(df.to_csv, "patient_phenotypes_v1.csv", index=False)


@app.callback(
    Output("dl-evidence", "data"),
    Input("btn-dl-evidence", "n_clicks"),
    State("store-evidence", "data"),
    prevent_initial_call=True,
)
def download_evidence(n, rows):
    if not rows:
        return no_update
    df = pd.DataFrame(rows)
    return dcc.send_data_frame(df.to_csv, "extraction_evidence.csv", index=False)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)

