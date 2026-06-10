# 🧬 phenotyper_dash

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![spaCy](https://img.shields.io/badge/spaCy-3.8-orange.svg)
![medspaCy](https://img.shields.io/badge/medspaCy-1.3.1-green.svg)
![Dash](https://img.shields.io/badge/Dash-Plotly-black.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)
![Status](https://img.shields.io/badge/status-Demo%20Hardened-success.svg)

A **local, fully offline Dash application** for extracting **breast cancer phenotypes** from free-text clinical notes using **rule-based NLP with spaCy + medspaCy** and transparent, auditable evidence tracking.

This project intentionally prioritizes **determinism, explainability, traceability, and privacy-safe local execution** over black-box prediction.

> ⚠️ **Demo safety warning:** This MVP is intended for **synthetic or de-identified notes only**. Do **not** upload real patient records or PHI during demos or testing.

---

## 🚀 What this app does

* Ingests **multiple `.txt` clinical notes per patient**
* Optionally uses a mapping CSV to assign notes to patients
* Extracts structured breast cancer phenotypes:

  * ER / PR status and percentages
  * HER2 IHC, FISH, and final status
  * Ki-67
  * Histology
  * Grade
  * Metastatic disease status
  * Clinical and pathologic stage
* Aggregates note-level findings into **one patient-level phenotype row**
* Preserves **all evidence mentions**, including:

  * Source note
  * Patient ID
  * Extracted field and value
  * Text snippet
  * Confidence score
  * Negation and uncertainty flags
* Runs optional validation against gold labels
* Exports patient, evidence, cohort-friendly, validation, and review CSVs
* Can download all outputs as a ZIP bundle with a README manifest
* Runs **entirely locally**:

  * No APIs
  * No cloud NLP calls
  * No LLM calls

---

## 🧠 Architecture overview

```text
Clinical notes (.txt)
        |
        v
[ spaCy + medspaCy ]
  - Rule-based NER
  - ConText negation / uncertainty detection
        |
        v
[ Normalization layer ]
  - Percent parsing
  - HER2 reconciliation
  - Histology normalization
  - Grade normalization
  - Stage normalization
        |
        v
[ Aggregation layer ]
  - Note-type precedence
  - Evidence-aware selection
  - Deterministic conflict resolution
        |
        v
[ Dash UI ]
  - Patient phenotype table
  - Evidence table with snippets
  - Validation against gold labels
  - Export bundle
```

---

## ✅ Demo-hardening features

The app has been hardened for first-user demo testing.

Current demo-safety and usability improvements include:

* Visible **no-PHI / synthetic-data-only warning**
* Guided demo checklist
* Sample demo data loader
* spaCy/scispaCy model availability preflight
* Loading indicators around extraction and validation outputs
* Per-note exception handling so one bad note does not crash the entire run
* Early file-format feedback for `.txt` notes and `.csv` mapping/gold files
* Cleaner patient phenotype column ordering
* Summary cards for:

  * Notes processed
  * Patients
  * Evidence mentions
  * Core fields populated
  * Validation fields evaluated
* Clear empty-download feedback
* Downloadable ZIP bundle with output manifest
* Demo-safe run mode with debug off by default

---

## 🛠 Installation

### Local virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm

python app.py
```

Then open:

```text
http://127.0.0.1:8050
```

---

## 🐳 Docker

```bash
docker build -t phenotyper-dash .
docker run -p 8050:8050 phenotyper-dash
```

Then open:

```text
http://127.0.0.1:8050
```

For demo/deployment use, the app should run with debug disabled.

---

## 📦 Apptainer / Singularity

```bash
apptainer build phenotyper.sif phenotyper.def
apptainer run phenotyper.sif
```

Then open the app in the browser using the exposed local port.

---

## 🧪 How to use the app

### 1. Load sample demo data or upload notes

For a quick demo, click:

```text
Load sample demo data
```

This loads local synthetic/de-identified sample notes, mapping, and gold labels from the app’s `sample_data/` folder.

You can also manually upload one or more `.txt` files containing clinical notes such as:

* Pathology reports
* Oncology consults
* Radiology reports
* Progress notes

---

### 2. Upload a mapping CSV

A mapping CSV lets you control patient identity and note metadata.

Supported columns:

| Column                  | Required | Description                     |
| ----------------------- | -------: | ------------------------------- |
| `note_id` OR `filename` |        ✅ | Links row to uploaded file      |
| `patient_id`            |        ✅ | Patient identifier              |
| `note_date`             |        ❌ | Used for precedence             |
| `note_type`             |        ❌ | Used for confidence and ranking |

If no metadata is available, defaults are:

```text
note_id = filename stem
patient_id = filename stem
note_type = Unknown
```

---

### 3. Select template and NLP model

The app detects available phenotype templates and installed spaCy/scispaCy models.

Before extraction, the app checks that the selected NLP model can actually be loaded. If the model is missing, extraction is blocked with a clear message instead of failing mid-run.

Default model:

```text
en_core_web_sm
```

---

### 4. Run extraction

Click:

```text
Run extraction
```

The app will:

* Process each uploaded note
* Extract structured phenotype fields
* Record evidence mentions and snippets
* Track negation and uncertainty
* Aggregate findings to patient level
* Continue processing other notes if one note fails

---

### 5. Review patient phenotypes

The patient phenotype table contains one row per patient.

Core fields are shown first, including:

* `patient_id`
* `er_status`
* `er_percent`
* `pr_status`
* `pr_percent`
* `her2_ihc_score`
* `her2_fish`
* `her2_final_status`
* `ki67_percent`
* `histology`
* `grade`
* `metastatic_disease`
* `stage_clinical`
* `stage_path`

Source and confidence columns are also preserved for auditability.

---

### 6. Review evidence

Select a patient row to inspect evidence mentions.

The evidence table includes:

* Patient ID
* Note ID
* Note date
* Note type
* Extracted field
* Extracted value
* Rule label
* Confidence score
* Negation flag
* Uncertainty flag
* Supporting text snippet

This makes each phenotype value traceable back to the original note text.

---

### 7. Run validation

Upload or load a gold labels CSV, then click:

```text
Run validation
```

Gold label columns should use the format:

```text
<field>__gold
```

Examples:

```text
er_status__gold
pr_status__gold
her2_final_status__gold
ki67_percent__gold
grade__gold
stage_path__gold
```

The app outputs:

* Field-level validation metrics
* Prediction-vs-gold review table
* Match flags for each evaluated field

---

## 📊 Outputs

The app can export individual CSV files or all outputs as a ZIP bundle.

Included outputs:

| File                         | Description                                                 |
| ---------------------------- | ----------------------------------------------------------- |
| `patient_phenotypes_v1.csv`  | One row per patient with aggregated phenotype values        |
| `extraction_evidence.csv`    | All evidence mentions with snippets and provenance          |
| `cohort_friendly_export.csv` | Long-format phenotype export suitable for cohort workflows  |
| `validation_metrics.csv`     | Field-level validation metrics                              |
| `prediction_vs_gold.csv`     | Row-level prediction-vs-gold comparison                     |
| `README_bundle.txt`          | Output manifest with timestamp, app version, and row counts |

The ZIP bundle includes a generated UTC timestamp, app version, patient count, evidence row count, and cohort row count.

---

## 🧮 Aggregation logic

For each phenotype field, the app uses deterministic rules:

1. Prefer values with **non-negated, non-uncertain evidence**
2. Prefer higher-priority note types:

   * Pathology / Addendum
   * Oncology consult
   * Radiology
   * Progress note
   * Unknown
3. Prefer newer notes if still tied
4. Fall back to the first non-empty value if no clean evidence exists

This makes the same input produce the same output every time.

---

## 🧬 HER2 final status rules

HER2 final status is reconciled deterministically:

* FISH overrides IHC
* IHC `3+` → `Positive`
* IHC `2+` → `Equivocal`
* IHC `0` or `1+` → `Negative`
* If no FISH/IHC result exists, explicit HER2 positive/negative/equivocal text may be used

---

## ⚠️ Known limitations

This is a rule-based MVP by design.

### NLP limitations

* No deep ML or transformer models
* Relies on curated rules and regex fallbacks
* May miss highly unusual phrasing
* May require additional rules for new note styles

### Clinical scope

* Breast cancer only
* Limited staging nuance
* No full TNM abstraction engine yet
* No treatment response inference
* No outcome prediction

### Data assumptions

* Text input only
* No OCR
* No scanned PDF processing
* Assumes reasonably clean clinical-note text

### Not intended to

* Replace manual chart review
* Make clinical decisions
* Serve as production clinical decision support
* Process real PHI in public demos

---

## 🎯 Why this design is intentional

The app is designed to be:

* **Deterministic:** same input → same output
* **Auditable:** every value is traceable to evidence text
* **Privacy-safe:** runs locally and offline
* **Extensible:** new rules and schemas can be added incrementally
* **Demo-friendly:** sample data flow is guided and repeatable
* **Research-oriented:** useful for preprocessing, abstraction support, and cohort discovery prototyping

Well-suited use cases include:

* Research preprocessing
* Cohort discovery
* Chart abstraction QA
* Phenotyping pipeline prototyping
* Rule-first extraction experiments
* Local/offline NLP demos

---

## 🔮 Future directions

Possible next steps:

* Add more breast cancer note formats and edge-case rules
* Improve TNM parsing
* Add multi-cancer schemas
* Add optional scispaCy biomedical models
* Add targeted ML only where rules fail
* Add genotype join keys such as `ERBB2`, `ESR1`, and `PGR`
* Add read-only authenticated deployment mode
* Add reviewer correction export/import loop
* Add richer export manifest with template/model/run metadata

---

## 📌 Status

Current status:

```text
MVP stable.
Demo hardened.
Schema locked.
Containerized.
Offline/local-first.
Ready for friendly first-user testing with synthetic or de-identified data.
```
