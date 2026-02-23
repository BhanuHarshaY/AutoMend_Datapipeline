"""
Airflow DAG for Glaive Function Calling v2 Data Pipeline.
Orchestrates the full pipeline from acquisition to bias detection.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
import logging
import sys
import os
sys.path.insert(0, "/opt/airflow/scripts")
os.environ["PYTHONPATH"] = "/opt/airflow/scripts:" + os.environ.get("PYTHONPATH", "")

logger = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner":            "bhanu_harsha",
    "depends_on_past":  False,
    "email_on_failure": False,
    "email_on_retry":   False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
}

with DAG(
    dag_id="glaive_data_pipeline",
    default_args=DEFAULT_ARGS,
    description="End to end data pipeline for Glaive Function Calling v2",
    schedule_interval="@weekly",
    start_date=days_ago(1),
    catchup=False,
    tags=["automend", "glaive", "data-pipeline"],
) as dag:

    def task_data_acquisition(**context):
        """
        Task 1: Stream and save 5000 records from HuggingFace.
        Reproducible acquisition with fixed random seed.
        """
        from data_acquisition import fetch_and_save
        from pathlib import Path

        output_file = Path("/opt/airflow/data/raw/glaive_raw.jsonl")
        count = fetch_and_save(
            sample_size=5000,
            output_file=output_file
        )
        logger.info("Acquired %d records", count)
        context["ti"].xcom_push(key="record_count", value=count)
        return count

    def task_preprocessing(**context):
        """
        Task 2: Parse, clean, extract features, and remap to ChatML format.
        """
        from preprocessing import run_preprocessing
        from pathlib import Path

        df = run_preprocessing(
            raw_file=Path("/opt/airflow/data/raw/glaive_raw.jsonl"),
            output_file=Path("/opt/airflow/data/processed/glaive_processed.jsonl"),
        )
        logger.info("Preprocessing complete. Shape: %s", df.shape)
        context["ti"].xcom_push(key="processed_count", value=len(df))
        return len(df)

    def task_schema_validation(**context):
        """
        Task 3: Validate processed data against Great Expectations schema.
        Fails pipeline if critical expectations are not met.
        """
        import json
        from schema_validation import load_processed_data, run_validation
        from pathlib import Path

        df      = load_processed_data(
            Path("/opt/airflow/data/processed/glaive_processed.jsonl")
        )
        results = run_validation(df)

        passed  = sum(1 for v in results.values() if v)
        failed  = sum(1 for v in results.values() if not v)
        total   = len(results)

        logger.info("Validation: %d/%d passed", passed, total)

        if failed > 0:
            failed_checks = [k for k, v in results.items() if not v]
            raise ValueError(
                f"Schema validation failed: {failed} checks failed: {failed_checks}"
            )

        context["ti"].xcom_push(key="validation_passed", value=passed)
        return passed

    def task_anomaly_detection(**context):
        """
        Task 4: Detect data anomalies and send Slack alerts if found.
        Does not fail pipeline but logs anomalies for monitoring.
        """
        from anomaly_detection import run_anomaly_detection
        from pathlib import Path

        report = run_anomaly_detection(
            filepath=Path("/opt/airflow/data/processed/glaive_processed.jsonl")
        )

        logger.info(
            "Anomaly detection: %d anomalies found",
            report["anomalies_found"]
        )
        context["ti"].xcom_push(
            key="anomalies_found",
            value=report["anomalies_found"]
        )
        return report["anomalies_found"]

    def task_bias_detection(**context):
        """
        Task 5: Detect representation bias across data slices.
        Logs findings and saves bias report for documentation.
        """
        from bias_detection import run_bias_detection
        from pathlib import Path

        report = run_bias_detection(
            filepath=Path("/opt/airflow/data/processed/glaive_processed.jsonl")
        )

        logger.info(
            "Bias detection: %d findings across %d slices",
            report["findings_count"],
            report["slices_analyzed"]
        )
        context["ti"].xcom_push(
            key="bias_findings",
            value=report["findings_count"]
        )
        return report["findings_count"]

    def task_dvc_versioning(**context):
        """
        Task 6: Version control processed data files with DVC.
        Tracks raw and processed datasets for reproducibility.
        """
        import subprocess

        files_to_track = [
            "/opt/airflow/data/raw/glaive_raw.jsonl",
            "/opt/airflow/data/processed/glaive_processed.jsonl",
            "/opt/airflow/data/processed/glaive_chatml.jsonl",
        ]

        for filepath in files_to_track:
            result = subprocess.run(
                ["dvc", "add", filepath],
                capture_output=True,
                text=True,
                cwd="/opt/airflow"
            )
            if result.returncode == 0:
                logger.info("DVC tracked: %s", filepath)
            else:
                logger.warning("DVC tracking failed for %s: %s", filepath, result.stderr)

        return "DVC versioning complete"

    def task_pipeline_summary(**context):
        """
        Task 7: Log final pipeline summary pulling XCom values from all tasks.
        """
        ti = context["ti"]

        record_count      = ti.xcom_pull(task_ids="data_acquisition",   key="record_count")
        processed_count   = ti.xcom_pull(task_ids="preprocessing",      key="processed_count")
        validation_passed = ti.xcom_pull(task_ids="schema_validation",  key="validation_passed")
        anomalies_found   = ti.xcom_pull(task_ids="anomaly_detection",  key="anomalies_found")
        bias_findings     = ti.xcom_pull(task_ids="bias_detection",     key="bias_findings")

        summary = {
            "pipeline":         "Glaive Function Calling v2",
            "run_date":         str(datetime.now()),
            "record_count":     record_count,
            "processed_count":  processed_count,
            "validation_passed": validation_passed,
            "anomalies_found":  anomalies_found,
            "bias_findings":    bias_findings,
            "status":           "SUCCESS",
        }

        logger.info("Pipeline Summary: %s", summary)
        print("\n" + "=" * 50)
        print("       PIPELINE SUMMARY")
        print("=" * 50)
        for k, v in summary.items():
            print(f"  {k:<25} {v}")
        print("=" * 50)

        return summary

    t1_acquisition = PythonOperator(
        task_id="data_acquisition",
        python_callable=task_data_acquisition,
        provide_context=True,
    )

    t2_preprocessing = PythonOperator(
        task_id="preprocessing",
        python_callable=task_preprocessing,
        provide_context=True,
    )

    t3_schema_validation = PythonOperator(
        task_id="schema_validation",
        python_callable=task_schema_validation,
        provide_context=True,
    )

    t4_anomaly_detection = PythonOperator(
        task_id="anomaly_detection",
        python_callable=task_anomaly_detection,
        provide_context=True,
    )

    t5_bias_detection = PythonOperator(
        task_id="bias_detection",
        python_callable=task_bias_detection,
        provide_context=True,
    )

    t6_dvc_versioning = PythonOperator(
        task_id="dvc_versioning",
        python_callable=task_dvc_versioning,
        provide_context=True,
    )

    t7_summary = PythonOperator(
        task_id="pipeline_summary",
        python_callable=task_pipeline_summary,
        provide_context=True,
    )

    t1_acquisition >> t2_preprocessing >> t3_schema_validation >> t4_anomaly_detection >> t5_bias_detection >> t6_dvc_versioning >> t7_summary