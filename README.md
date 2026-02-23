# AutoMend Data Pipeline
### Glaive Function Calling v2 — Syntax Layer Pipeline
IE7374 MLOps | Northeastern University

---

## Project Overview

This repository contains the data pipeline for the AutoMend MLOps platform, specifically handling the Glaive Function Calling v2 dataset. AutoMend is a self-healing MLOps platform,  "Zapier for MLOps,"  that autonomously remediates production ML incidents through event-driven workflows.

The Glaive Function Calling v2 dataset serves as the Syntax Layer in AutoMend's training strategy. It teaches the Generative Architect (fine-tuned Llama-3-8B) the fundamental skill of converting natural language requests into structured, valid JSON outputs. This pipeline handles everything from raw data acquisition through preprocessing, schema validation, anomaly detection, bias analysis, and final ChatML formatting — all orchestrated through Apache Airflow.

The pipeline produces a ready-to-use ChatML JSONL file that feeds directly into AutoMend's LLM fine-tuning workflow.

---

## Pipeline Architecture

```
data_acquisition
      |
preprocessing
      |
schema_validation
      |
anomaly_detection
      |
bias_detection
      |
dvc_versioning
      |
pipeline_summary
```

All 7 tasks run sequentially. Each task passes results to the next via Airflow XCom. If schema validation fails, the pipeline halts immediately rather than propagating bad data downstream.

---

## Project Structure

```
AutoMend_Datapipeline/
├── dags/
│   └── glaive_pipeline_dag.py       # Airflow DAG, 7 PythonOperator tasks
├── scripts/
│   ├── data_acquisition.py          # HuggingFace streaming, 5000 record sample
│   ├── preprocessing.py             # Parsing, feature engineering, ChatML remapping
│   ├── schema_validation.py         # Great Expectations, 30 checks
│   ├── anomaly_detection.py         # Threshold-based checks, Slack alerting
│   └── bias_detection.py            # Data slicing across 6 dimensions
├── tests/
│   ├── test_acquisition.py          # 4 unit tests
│   ├── test_preprocessing.py        # 20 unit tests
│   ├── test_schema_validation.py    # 4 unit tests
│   ├── test_anomaly_detection.py    # 11 unit tests
│   └── test_bias_detection.py       # 16 unit tests
├── data/
│   ├── raw/
│   │   └── glaive_raw.jsonl.dvc     # DVC pointer, 5000 raw records
│   └── processed/
│       ├── glaive_processed.jsonl.dvc   # DVC pointer, 5000 enriched records
│       ├── glaive_chatml.jsonl          # Final ChatML output for fine-tuning
│       └── validation/
│           ├── validation_report.json   # GE results, 30/30 checks
│           ├── anomaly_report.json      # 6 anomaly check results
│           └── bias_report.json         # 6 slice analysis, 3 findings
├── logs/
├── docker-compose.yml               # Airflow cluster, 6 services
├── requirements.txt
├── .dvc/
├── .dvcignore
└── .env                             # Not committed, see Environment Variables
```

---

## Dataset

**Name:** Glaive Function Calling v2  
**Source:** https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2  
**License:** cc-by-sa-4.0  
**Original size:** 113,000+ examples, approximately 2.5 GB  
**Pipeline sample:** 5,000 records (streaming, no full download)  
**Format:** JSONL with system, user, and assistant turns  

**Role in AutoMend:** The dataset teaches the Generative Architect how to produce valid JSON function calls from natural language. Even though the functions in Glaive (weather, exchange rates, etc.) are unrelated to MLOps, the structural patterns — how JSON brackets open and close, how arguments are typed, how multi-turn context is maintained — directly transfer to AutoMend's remediation workflow generation.

---

## Prerequisites

- Python 3.10
- Docker Desktop (for Airflow)
- Git
- A HuggingFace account (free) for dataset streaming
- A Slack workspace and bot token (for anomaly alerts, optional but recommended)

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/BhanuHarshaY/AutoMend_Datapipeline.git
cd AutoMend_Datapipeline
```

### 2. Create a virtual environment with Python 3.10

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Create your .env file

Create a `.env` file in the project root. This file is never committed to git.

```env
HF_TOKEN=your_huggingface_token_here
SLACK_BOT_TOKEN=xoxb-your-slack-bot-token
SLACK_CHANNEL=all-automend
AIRFLOW_UID=50000
```

To get your HuggingFace token, go to https://huggingface.co/settings/tokens and create a read token.

To get your Slack bot token, create a Slack app at https://api.slack.com/apps, add the `chat:write` scope, install it in your workspace, and copy the Bot User OAuth Token.

### 4. Run the pipeline locally (without Airflow)

Each script can be run independently for testing:

```bash
python scripts/data_acquisition.py
python scripts/preprocessing.py
python scripts/schema_validation.py
python scripts/anomaly_detection.py
python scripts/bias_detection.py
```

### 5. Run the full test suite

```bash
pytest tests/ -v --cov=scripts --cov-report=term-missing
```

Expected output: 55 tests passing across all 5 test files.

---

## Running with Airflow (Docker)

### Step 1 — Initialize Airflow

Run this once to create the database and default admin user:

```bash
docker compose up airflow-init
```

Wait until you see:
```
airflow-init exited with code 0
```

### Step 2 — Start all services

```bash
docker compose up -d
```

This starts 6 containers: postgres, redis, airflow-webserver, airflow-scheduler, airflow-worker, and airflow-triggerer.

### Step 3 — Verify all containers are healthy

```bash
docker compose ps
```

All services should show `healthy` or `running` status.

### Step 4 — Open the Airflow UI

Go to http://localhost:8080 and log in with:

```
Username: airflow
Password: airflow
```

### Step 5 — Trigger the pipeline

1. Find the `glaive_data_pipeline` DAG in the list
2. Click on it to open the DAG view
3. Click the play button in the top right corner
4. Select "Trigger DAG."
5. Watch the Graph view — all 7 tasks should turn green sequentially

Total runtime is approximately 5 to 8 minutes, dominated by the HuggingFace streaming step.

### Step 6 — Stop all services

```bash
docker compose down
```

To fully reset, including database volumes:

```bash
docker compose down --volumes --remove-orphans
```

---

## Data Versioning with DVC

DVC tracks large data files separately from Git, so the repository stays lightweight.

Initialize DVC tracking after running the pipeline:

```bash
dvc add data/raw/glaive_raw.jsonl
dvc add data/processed/glaive_processed.jsonl
dvc add data/processed/glaive_chatml.jsonl
git add data/raw/glaive_raw.jsonl.dvc data/processed/glaive_processed.jsonl.dvc data/processed/glaive_chatml.jsonl.dvc
git commit -m "data: version pipeline outputs with DVC."
```

Useful DVC commands:

| Command | Purpose |
|---|---|
| `dvc status` | Show which tracked files have changed |
| `dvc repro` | Re-run pipeline stages that are out of date |
| `dvc push` | Upload data to remote storage |
| `dvc pull` | Download data from remote storage |
| `dvc dag` | Print the pipeline dependency graph |

To reproduce the exact dataset from scratch on a new machine:

```bash
git clone https://github.com/BhanuHarshaY/AutoMend_Datapipeline.git
cd AutoMend_Datapipeline
pip install -r requirements.txt
python scripts/data_acquisition.py
```

The `RANDOM_SEED = 42` in `data_acquisition.py` ensures the same 5000 records are sampled every time.

---

## Pipeline Components

### Data Acquisition (`scripts/data_acquisition.py`)

Streams the Glaive Function Calling v2 dataset from HuggingFace using `streaming=True` so the full 2.5 GB is never downloaded locally. Collects exactly 5000 records using a fixed random seed of 42 for reproducibility and saves them as JSONL to `data/raw/glaive_raw.jsonl`.

Key configuration:

```python
DATASET_NAME = "glaiveai/glaive-function-calling-v2"
SAMPLE_SIZE  = 5000
RANDOM_SEED  = 42
```

### Preprocessing (`scripts/preprocessing.py`)

Parses each raw record and extracts the following features:

| Feature | Description |
|---|---|
| `num_turns` | Number of USER turns in the conversation |
| `num_calls` | Number of function calls made by the assistant |
| `complexity_tier` | none, simple, moderate, complex, or malformed |
| `has_parallel` | Whether multiple function calls appear in one turn |
| `has_malformed` | Whether any function call JSON was unparseable |
| `num_defined_functions` | Number of functions defined in the system prompt |
| `defined_function_names` | List of function names from the system prompt |
| `function_signatures` | Full parameter definitions extracted from system prompt |
| `has_error_handling` | Whether the conversation contains error-related language |
| `has_function_error_response` | Whether a function response contains an error |
| `has_conditional_error` | Whether the assistant handles errors conditionally |

The script also handles Glaive's non-standard JSON format where `arguments` is a single-quoted string rather than a proper JSON object, and correctly parses array-style arguments used in some numeric functions.

After processing, all 5000 records are remapped to ChatML format for fine-tuning:

```json
{
  "messages": [
    {"role": "system", "content": "You are AutoMend... Available Tools: {...}"},
    {"role": "user", "content": "natural language request"},
    {"role": "assistant", "content": "{\"workflow\": {\"steps\": [...]}}"}
  ],
  "complexity_tier": "simple",
  "num_turns": 2,
  "num_calls": 1
}
```

This is the standard format expected by HuggingFace TRL for supervised fine-tuning.

### Schema Validation (`scripts/schema_validation.py`)

Runs 30 Great Expectations checks on the processed DataFrame:

- 15 checks verifying all required columns exist
- 4 checks verifying no nulls in critical columns
- 3 checks verifying numeric ranges are sensible
- 1 check verifying complexity tier values are within the expected set
- 5 checks verifying boolean columns contain only True or False
- 1 check verifying the chat field is not empty
- 1 check verifying the dataset row count is within expected bounds

All 30 checks must pass for the pipeline to continue. Results are saved to `data/processed/validation/validation_report.json`.

### Anomaly Detection (`scripts/anomaly_detection.py`)

Runs 6 threshold-based checks against the processed dataset:

| Check | Threshold | Rationale |
|---|---|---|
| Malformed call rate | Less than 5% | High malformed rate indicates parsing failure |
| None complexity rate | Less than 60% | Too many no-call records suggests data skew |
| Minimum record count | At least 4000 | Catches partial acquisition failures |
| Average turn count | Less than 10 | Extremely long conversations are anomalous |
| Average call count | Less than 5 | Very high call counts are anomalous |
| Defined function coverage | At least 30% | Low coverage suggests system prompt parsing failure |

When anomalies are detected the script sends a formatted alert to the configured Slack channel. Results are saved to `data/processed/validation/anomaly_report.json`.

### Bias Detection (`scripts/bias_detection.py`)

Slices the dataset across 6 dimensions and flags any slice representing less than 5% of total records as underrepresented. Since Glaive contains no demographic features, meaningful proxies are used instead:

| Slice | Rationale |
|---|---|
| `complexity_tier` | Tests whether fine-tuning data covers all complexity levels equally |
| `turn_bucket` | Tests whether single-turn and multi-turn conversations are balanced |
| `call_bucket` | Tests whether varying call counts are adequately represented |
| `has_error_handling` | Tests whether error handling examples are represented |
| `has_parallel` | Tests whether parallel call examples are represented |
| `has_defined_functions` | Tests whether function-defined vs undefined conversations are balanced |

**Findings from current dataset (5000 records):**

Three underrepresented slices were detected:

1. `complexity_tier = malformed` — 7 records (0.14%), HIGH severity. These are genuine parsing edge cases. Mitigation: random oversampling to bring above 5% threshold, or exclude from training entirely since malformed examples teach bad syntax.

2. `turn_bucket = long` — 117 records (2.34%), MEDIUM severity. Long multi-turn conversations are naturally rare in the dataset. Mitigation: apply sample weights during fine-tuning to up-weight these examples, or collect additional long-conversation examples from the full 113K dataset.

3. `call_bucket = many_calls` — 41 records (0.82%), HIGH severity. Complex multi-call scenarios are rare. Mitigation: random oversampling. Note that for AutoMend this slice is particularly important since remediation workflows often require chained tool calls.

Results are saved to `data/processed/validation/bias_report.json`.

---

## Tracking, Logging, and Alerts

All scripts use Python's `logging` module with a consistent format:

```
2026-02-23 01:00:57,000 | INFO | Loading raw data from ...
2026-02-23 01:00:57,100 | INFO | Processed 1000 / 5000 records
```

Logs are written to stdout and captured by Airflow's task logging system. Each task's logs are accessible through the Airflow UI by clicking on any task node and selecting "Log".

Slack alerts are triggered by the anomaly detection task whenever any check exceeds its threshold. The alert message includes the check name, observed value, threshold, and a recommendation. To enable alerts, configure `SLACK_BOT_TOKEN` and `SLACK_CHANNEL` in your `.env` file and invite the bot to the target channel using `/invite @your-bot-name`.

---

## Pipeline Flow Optimization

The current pipeline runs sequentially. This is appropriate because each stage depends on the output of the previous stage. The Gantt chart in Airflow (DAG view, Gantt tab) shows the wall-clock time per task.

From a typical run:

| Task | Approximate Duration |
|---|---|
| data_acquisition | 4 to 6 minutes (network-bound) |
| preprocessing | 15 seconds |
| schema_validation | 5 seconds |
| anomaly_detection | 3 seconds |
| bias_detection | 3 seconds |
| dvc_versioning | 2 seconds |
| pipeline_summary | 1 second |

The data acquisition step dominates total runtime. If this becomes a bottleneck, the sample size can be reduced or the raw JSONL can be pre-cached locally and the acquisition step skipped on subsequent runs using Airflow's `skip` mechanism.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `HF_TOKEN` | Yes | HuggingFace read token for dataset streaming |
| `SLACK_BOT_TOKEN` | No | Slack bot token starting with xoxb- |
| `SLACK_CHANNEL` | No | Slack channel name without # symbol |
| `AIRFLOW_UID` | Yes (Docker) | Unix user ID, use output of `id -u` |

---

## Reproducibility

The pipeline is fully reproducible:

- `RANDOM_SEED = 42` in `data_acquisition.py` ensures identical 5000-record samples
- `requirements.txt` pins all package versions explicitly
- DVC `.dvc` pointer files allow exact dataset versions to be pulled by any team member
- Docker Compose ensures identical Airflow and Python environments across machines

To reproduce results on a new machine:

```bash
git clone https://github.com/BhanuHarshaY/AutoMend_Datapipeline.git
cd AutoMend_Datapipeline
python3.10 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then fill in your tokens
python scripts/data_acquisition.py
python scripts/preprocessing.py
python scripts/schema_validation.py
python scripts/anomaly_detection.py
python scripts/bias_detection.py
```

---

## Test Coverage

Run the full test suite with coverage:

```bash
pytest tests/ -v --cov=scripts --cov-report=term-missing
```

| Test File | Tests | What it covers |
|---|---|---|
| `test_acquisition.py` | 4 | File creation, JSONL validity, required fields, sample size |
| `test_preprocessing.py` | 20 | Function call extraction, signature parsing, error detection, ChatML format |
| `test_schema_validation.py` | 4 | Valid schema passes, invalid complexity fails, dict return type |
| `test_anomaly_detection.py` | 11 | Each check with healthy and anomalous fixtures, result key presence |
| `test_bias_detection.py` | 16 | Slice feature engineering, proportion sums, severity classification, mitigation strategies |

---

## Team

This pipeline covers the Glaive Function Calling v2 dataset component of the AutoMend project.

Full team: Bhanu Harsha Yanamadala, Jennisha Christina Martin, Mohammed Ahnaf Tajwar, Raghav Jadia, Sanjana Satish Menon, Sri Ram Sathiya Narayanan

Repository: https://github.com/BhanuHarshaY/AutoMend_Datapipeline  
Full project repository: https://github.com/BhanuHarshaY/AutoMend  
Course: IE7374 MLOps, Northeastern University
