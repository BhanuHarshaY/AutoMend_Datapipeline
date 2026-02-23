
---

# AutoMend Data Pipeline

An end-to-end MLOps data pipeline built to process the [Glaive Function Calling v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) dataset. This pipeline handles data streaming, parsing, feature engineering, ChatML formatting, schema validation, anomaly alerting via Slack, and representation bias detection — orchestrated with Apache Airflow and versioned with DVC.

---

## Pipeline Overview

```text
data_acquisition
       │
       ▼
 preprocessing
       │
       ▼
schema_validation
       │
       ▼
anomaly_detection
       │
       ▼
 bias_detection
       │
       ▼
 dvc_versioning
       │
       ▼
pipeline_summary

```

**Dataset Focus:** 5,000 conversational samples engineered for LLM function calling and tool use.

The pipeline sequentially extracts unformatted text blocks, safely parses inner JSON arguments, standardizes them into the conversational `ChatML` format for Llama-3 fine-tuning, and runs rigorous automated QA checks.

---

## Project Structure

```text
AutoMend_Datapipeline/
├── dags/                        # Airflow orchestration logic
│   └── glaive_pipeline_dag.py   # Main Directed Acyclic Graph defining task execution order
├── scripts/
│   ├── data_acquisition.py      # Streams dataset from HuggingFace
│   ├── preprocessing.py         # Parses nested JSON, engineers features, converts to ChatML
│   ├── schema_validation.py     # Great Expectations structural checks
│   ├── anomaly_detection.py     # Custom threshold monitoring and Slack alerts
│   └── bias_detection.py        # Data slicing for representation bias analysis
├── tests/                       # Complete pytest suite mapping 1:1 with pipeline scripts
├── data/                        # Local output directory (mounted to Docker)
│   ├── raw/                     # Original streamed JSONL
│   └── processed/               # Cleaned data, ChatML outputs, and validation reports
├── docker-compose.yml           # Containerized Airflow + PostgreSQL + Redis environment
├── requirements.txt             # Pinned Python 3.8-compatible dependencies
└── .gitignore / .dvcignore      # Version control exclusions

```

---

## Quickstart

### Prerequisites

* [Docker Desktop](https://www.docker.com/products/docker-desktop/)
* Python 3.8+ (for local testing)

### 1. Clone the repo and set up the environment

```bash
git clone https://github.com/BhanuHarshaY/AutoMend_Datapipeline.git
cd AutoMend_Datapipeline

# Create a virtual environment and install pinned dependencies
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt

```

### 2. Configure Environment Variables

Create a `.env` file in the project root to enable Slack alerting during anomaly detection. This file is excluded from Git.

```env
# Optional: Set up Slack alerts
SLACK_BOT_TOKEN=xoxb-your-bot-token-here
SLACK_CHANNEL=#automend-alerts

# Airflow user configuration
AIRFLOW_UID=50000

```

### 3. Run the pipeline via Airflow (Docker)

**Step 1 — Wipe existing state and start the cluster:**

```bash
docker compose down -v
docker compose up -d --force-recreate

```

*Note: The initial boot may take a few minutes as the Airflow webserver installs heavy data science dependencies (pandas, scikit-learn) directly from `requirements.txt`.*

**Step 2 — Open the Airflow UI:**
Go to **http://localhost:8080** and log in:

* **Username:** `airflow`
* **Password:** `airflow`

**Step 3 — Run the pipeline:**

1. Locate the `glaive_data_pipeline` DAG.
2. Unpause it using the toggle on the left.
3. Click the **▶ Play** button and select **Trigger DAG**.
4. Navigate to the **Grid** or **Graph** view to monitor task execution.

**To safely stop the cluster:**

```bash
docker compose down

```

### 4. Run tests locally

```bash
pytest tests/ -v

```

---

## Tracking, Logging & Anomaly Alerts

**Logging:**
Standard Python `logging` is utilized across all modules. Logs are written to `stdout` and captured directly by Airflow's task logs for easy debugging in the UI.

**Anomaly Alerts (`anomaly_detection.py`):**
Instead of relying solely on Airflow task failures, the pipeline implements business-logic anomaly detection. It evaluates the parsed dataset against specific thresholds:

* Malformed JSON extraction rate > 5%
* Average conversational turns > 10
* Datasets returning fewer than 4,000 records

If a threshold is breached, the pipeline uses `slack-sdk` to fire a formatted summary directly to a designated Slack channel, allowing the engineering team to intervene before corrupted data reaches the training phase.

---

## Data Bias Detection & Mitigation

### What bias means for LLM Tool-Use datasets

Because the Glaive dataset consists of synthetic AI conversations, traditional demographic data slicing (age, gender, race) is not applicable. Instead, "bias" in this context refers to **representational imbalance in conversational complexity**.

If an LLM is fine-tuned on data where 95% of queries only require one simple function call, the resulting model will be biased toward simple actions and fail at complex, multi-step reasoning.

### Data Slicing Approach (`bias_detection.py`)

To ensure a balanced fine-tuning dataset, `preprocessing.py` engineers custom categorical features that the bias detector then evaluates:

| Slice | What it measures |
| --- | --- |
| `complexity_tier` | Balances "simple" 1-arg calls vs "complex" parallel calls |
| `turn_bucket` | Checks ratio of single-turn vs multi-turn conversations |
| `has_error_handling` | Ensures the model learns how to gracefully handle failed tool calls |

### Detection and Mitigation

Slices falling below a `5%` representation threshold trigger a warning in the validation logs. The pipeline automatically outputs mitigation strategies into `bias_report.json`, classifying imbalances by severity and recommending targeted oversampling (e.g., SMOTE) or adjusted loss weighting for the downstream training stage.

---

## Schema & Validation

`schema_validation.py` utilizes **Great Expectations (v0.18.12)** to run strict structural QA. The pipeline validates that:

* Core dataset fields (`system`, `chat`, `function_calls`) contain no nulls.
* Categorical mappings match exact sets (e.g., `complexity_tier` must be one of `['none', 'simple', 'moderate', 'complex', 'malformed']`).
* Bounding checks are respected (e.g., `num_turns` between 0 and 50).
* The final dataset maintains a row count between 4,000 and 6,000.

---

## Key Design Decisions

| Decision | Choice | Reason |
| --- | --- | --- |
| **Data Formatting** | ChatML Format | Standardizes the raw "USER/ASSISTANT" text into a normalized system/user/assistant JSON array optimized for Llama-3 instruction tuning. |
| **Dependency Pinning** | `pandas==2.0.3` & `scikit-learn==1.3.2` | The official Apache Airflow Docker image utilizes Python 3.8. Newer library versions dropped 3.8 support; these specific versions ensure total stability inside the containerized workers. |
| **Code Modularity** | Separated DAGs vs Scripts | The execution logic (`scripts/`) is completely decoupled from the orchestrator (`dags/`). This enables independent unit testing and allows DVC to run the scripts locally without booting the full Airflow UI. |
| **Type Hinting** | `typing.Optional` | Modern ` |

---

## Outputs

All outputs are automatically synced to your local machine via Docker volume mounts inside the `data/` directory.

| File | Description |
| --- | --- |
| `data/raw/glaive_raw.jsonl` | Original 5k record stream downloaded from HuggingFace. |
| `data/processed/glaive_processed.jsonl` | Cleaned records with engineered metrics (turn counts, complexity). |
| `data/processed/glaive_chatml.jsonl` | The final artifact: data formatted strictly to ChatML standards, ready for model fine-tuning. |
| `data/processed/validation/validation_report.json` | Pass/Fail metrics from the Great Expectations suite. |
| `data/processed/validation/anomaly_report.json` | Dataset quality metrics and threshold breaches. |
| `data/processed/validation/bias_report.json` | Representation statistics for conversation complexity slices. |
