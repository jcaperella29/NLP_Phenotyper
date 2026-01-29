# 🧬 phenotyper_dash

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![spaCy](https://img.shields.io/badge/spaCy-3.8-orange.svg)
![medspaCy](https://img.shields.io/badge/medspaCy-1.3.1-green.svg)
![Dash](https://img.shields.io/badge/Dash-Plotly-black.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)
![Status](https://img.shields.io/badge/status-MVP%20Stable-success.svg)

A **local, fully offline Dash application** for extracting **breast cancer phenotypes** from free-text clinical notes using **rule-based NLP (spaCy + medspaCy)** with transparent, auditable evidence tracking.

This project intentionally prioritizes **determinism, explainability, and traceability** over black-box prediction.

---

## 🚀 What this app does

- Ingests **multiple clinical notes per patient**
- Extracts structured breast cancer phenotypes:
  - ER / PR status and percentages
  - HER2 (IHC, FISH, final status)
  - Ki-67
  - Histology
  - Grade
  - Clinical and pathologic stage
- Aggregates note-level findings into **one patient-level row**
- Preserves **all evidence mentions**, including:
  - Source note
  - Text snippet
  - Confidence score
  - Negation / uncertainty flags
- Runs **entirely locally** (no APIs, no cloud, no LLM calls)

---Clinical notes (.txt)
|
v
[ spaCy + medspaCy ]

Rule-based NER

ConText (negation / uncertainty)
|
v
[ Normalization layer ]

Percent parsing

HER2 reconciliation

Histology / grade / stage normalization
|
v
[ Aggregation layer ]

Note-type precedence

Evidence-aware selection

Deterministic conflict resolution
|
v
Dash UI

Patient phenotype table

Evidence table with snippets


---

## 🛠 Installation

### Local (virtualenv)

``` in bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm

python app.py

Open it at  http://127.0.0.1:8050

###🐳 Docker
in bash
docker build -t phenotyper-dash .
docker run -p 8050:8050 phenotyper-dash


Then open it same as before.

###📦 Apptainer / Singularity

in bash

apptainer build phenotyper.sif phenotyper.def
apptainer run phenotyper.sif
then open it again in the standard way.

🧪 How to use the app
1️⃣ Upload notes

Upload one or more .txt files containing clinical notes such as:

Pathology

Oncology consults

Radiology reports

Progress notes

2️⃣ (Optional) Upload a mapping CSV

A mapping CSV lets you control patient identity and note metadata.

Supported columns:
🧪 How to use the app
1️⃣ Upload notes

Upload one or more .txt files containing clinical notes such as:

Pathology

Oncology consults

Radiology reports

Progress notes

2️⃣ (Optional) Upload a mapping CSV

A mapping CSV lets you control patient identity and note metadata.

Supported columns:
| Column                  | Required | Description                   |
| ----------------------- | -------- | ----------------------------- |
| `note_id` OR `filename` | ✅        | Links row to uploaded file    |
| `patient_id`            | ✅        | Patient identifier            |
| `note_date`             | ❌        | Used for precedence           |
| `note_type`             | ❌        | Used for confidence & ranking |


Column	Required	Description
note_id OR filename	✅	Links row to uploaded file
patient_id	✅	Patient identifier
note_date	❌	Used for precedence
note_type	❌	Used for confidence & ranking

If no mapping is provided, defaults are:

note_id = filename stem

patient_id = filename stem

note_type = Unknown

3️⃣ Run extraction

Click “Run extraction”.

The app will:

Process each note with spaCy / medspaCy

Extract structured fields

Record all evidence mentions

Aggregate results to patient level


📊 Outputs
Patient phenotype table

One row per patient

Deterministic values

Source note metadata

Confidence buckets

Evidence table

Every extracted mention

Original text snippet

Negation / uncertainty flags

Confidence score

Both tables can be exported as CSV.

🧮 Aggregation logic (important)

For each phenotype field:

Prefer values with non-negated, non-uncertain evidence

Prefer Pathology / Addendum notes over Consults

Prefer newer notes if still tied

Fall back to first non-empty value if no clean evidence exists

HER2 final status rules

FISH overrides IHC

IHC 3+ → Positive

IHC 2+ → Equivocal

IHC 0 / 1+ → Negative

⚠️ Known limitations (v1)

This is a rule-based MVP by design.

NLP limitations

No deep ML / transformer models

Relies on curated rules and patterns

May miss highly non-standard phrasing

Clinical scope

Breast cancer only

Limited staging nuance (no full TNM parsing)

No treatment response or outcome inference

Data assumptions

Text input only (.txt)

No OCR / scanned PDFs

Assumes reasonably clean clinical notes

Not intended to:

Replace manual chart review

Make clinical decisions

Serve as a production CDS system

🎯 Why this design is intentional

Deterministic: same input → same output

Auditable: every value traceable to text

Privacy-safe: runs fully offline

Extensible: easy to add new rules or targeted ML later

Well-suited for:

Research preprocessing

Cohort discovery

QA / abstraction support

Phenotyping pipeline prototyping

🔮 Future directions (optional)

Targeted ML only where rules fail (e.g. free-text histology)

Genotype join keys (ERBB2, ESR1, PGR)

TNM parsing

Multi-cancer schemas

Read-only deployments behind auth

📌 Status

MVP complete.
Schema locked.
Containerized.
Ready for iteration and extension.
## 🧠 Architecture overview

