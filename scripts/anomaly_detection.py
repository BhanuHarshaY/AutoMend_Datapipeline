"""
Anomaly Detection & Alerting for Glaive Function Calling v2
Detects data anomalies and triggers alerts via logging and Slack.
"""

import json
import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

#  Logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Config
PROCESSED_FILE = Path(__file__).resolve().parents[1] / "data" / "processed" / "glaive_processed.jsonl"
ANOMALY_DIR    = Path(__file__).resolve().parents[1] / "data" / "processed" / "validation"

# Thresholds
THRESHOLDS = {
    "max_malformed_pct":       0.05,   # >5% malformed is anomalous
    "max_none_complexity_pct": 0.60,   # >60% none complexity is anomalous
    "min_records":             4000,   # fewer than 4000 records is anomalous
    "max_avg_turns":           10.0,   # avg turns >10 is anomalous
    "max_avg_calls":            5.0,   # avg calls >5 is anomalous
    "min_defined_fn_pct":      0.30,   # <30% records with defined functions
}


#  Slack Alert 
def send_slack_alert(message: str) -> None:
    """Send anomaly alert to Slack if token is configured."""
    token   = os.getenv("SLACK_BOT_TOKEN")
    channel = os.getenv("SLACK_CHANNEL", "#automend-alerts")

    if not token or token == "your_slack_bot_token_here":
        logger.warning("Slack token not configured — skipping Slack alert")
        logger.warning("ALERT MESSAGE: %s", message)
        return

    try:
        from slack_sdk import WebClient
        client = WebClient(token=token)
        client.chat_postMessage(channel=channel, text=message)
        logger.info("Slack alert sent to %s", channel)
    except Exception as e:
        logger.error("Failed to send Slack alert: %s", e)


#  Anomaly Checks 
def check_malformed_rate(df: pd.DataFrame) -> dict:
    """Check if malformed record percentage exceeds threshold."""
    malformed_pct = df["has_malformed"].sum() / len(df)
    is_anomaly    = malformed_pct > THRESHOLDS["max_malformed_pct"]
    return {
        "check":      "malformed_rate",
        "value":      round(malformed_pct, 4),
        "threshold":  THRESHOLDS["max_malformed_pct"],
        "is_anomaly": is_anomaly,
        "message":    f"Malformed rate {malformed_pct:.2%} exceeds threshold {THRESHOLDS['max_malformed_pct']:.2%}"
                      if is_anomaly else "OK",
    }


def check_none_complexity_rate(df: pd.DataFrame) -> dict:
    """Check if too many records have no function calls."""
    none_pct   = (df["complexity_tier"] == "none").sum() / len(df)
    is_anomaly = none_pct > THRESHOLDS["max_none_complexity_pct"]
    return {
        "check":      "none_complexity_rate",
        "value":      round(none_pct, 4),
        "threshold":  THRESHOLDS["max_none_complexity_pct"],
        "is_anomaly": is_anomaly,
        "message":    f"None complexity rate {none_pct:.2%} exceeds threshold"
                      if is_anomaly else "OK",
    }


def check_record_count(df: pd.DataFrame) -> dict:
    """Check if dataset has minimum required records."""
    count      = len(df)
    is_anomaly = count < THRESHOLDS["min_records"]
    return {
        "check":      "record_count",
        "value":      count,
        "threshold":  THRESHOLDS["min_records"],
        "is_anomaly": is_anomaly,
        "message":    f"Record count {count} below minimum {THRESHOLDS['min_records']}"
                      if is_anomaly else "OK",
    }


def check_avg_turns(df: pd.DataFrame) -> dict:
    """Check if average turn count is within expected range."""
    avg_turns  = df["num_turns"].mean()
    is_anomaly = avg_turns > THRESHOLDS["max_avg_turns"]
    return {
        "check":      "avg_turns",
        "value":      round(avg_turns, 4),
        "threshold":  THRESHOLDS["max_avg_turns"],
        "is_anomaly": is_anomaly,
        "message":    f"Avg turns {avg_turns:.2f} exceeds threshold"
                      if is_anomaly else "OK",
    }


def check_avg_calls(df: pd.DataFrame) -> dict:
    """Check if average function call count is within expected range."""
    avg_calls  = df["num_calls"].mean()
    is_anomaly = avg_calls > THRESHOLDS["max_avg_calls"]
    return {
        "check":      "avg_calls",
        "value":      round(avg_calls, 4),
        "threshold":  THRESHOLDS["max_avg_calls"],
        "is_anomaly": is_anomaly,
        "message":    f"Avg calls {avg_calls:.2f} exceeds threshold"
                      if is_anomaly else "OK",
    }


def check_defined_functions_coverage(df: pd.DataFrame) -> dict:
    """Check if enough records have defined function signatures."""
    has_fn_pct = (df["num_defined_functions"] > 0).sum() / len(df)
    is_anomaly = has_fn_pct < THRESHOLDS["min_defined_fn_pct"]
    return {
        "check":      "defined_functions_coverage",
        "value":      round(has_fn_pct, 4),
        "threshold":  THRESHOLDS["min_defined_fn_pct"],
        "is_anomaly": is_anomaly,
        "message":    f"Defined function coverage {has_fn_pct:.2%} below threshold"
                      if is_anomaly else "OK",
    }


#  Main 
def run_anomaly_detection(
    filepath: Path = PROCESSED_FILE,
) -> dict:
    """Run all anomaly checks and trigger alerts if needed."""
    ANOMALY_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading processed data from %s", filepath)
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    logger.info("Loaded %d records", len(df))

    # Run all checks
    checks = [
        check_malformed_rate(df),
        check_none_complexity_rate(df),
        check_record_count(df),
        check_avg_turns(df),
        check_avg_calls(df),
        check_defined_functions_coverage(df),
    ]

    # Identify anomalies
    anomalies = [c for c in checks if c["is_anomaly"]]

    # Log results
    logger.info("--- Anomaly Detection Results ---")
    for check in checks:
        status = "🚨 ANOMALY" if check["is_anomaly"] else " OK"
        logger.info("%s | %s | value=%.4f | threshold=%.4f",
                    status, check["check"],
                    float(check["value"]),
                    float(check["threshold"]))

    # Send alerts for anomalies
    if anomalies:
        alert_msg = (
            f"🚨 *AutoMend Data Pipeline Alert*\n"
            f"Dataset: Glaive Function Calling v2\n"
            f"Anomalies detected: {len(anomalies)}\n\n"
        )
        for a in anomalies:
            alert_msg += f"• {a['check']}: {a['message']}\n"
        send_slack_alert(alert_msg)
    else:
        logger.info("No anomalies detected — pipeline healthy ")

    # Save anomaly report
    report = {
        "total_checks":     len(checks),
        "anomalies_found":  len(anomalies),
        "pipeline_healthy": len(anomalies) == 0,
        "checks":           checks,
    }
    report_path = ANOMALY_DIR / "anomaly_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=lambda x: bool(x) if hasattr(x, 'item') else str(x))
    logger.info("Anomaly report saved to %s", report_path)

    return report


if __name__ == "__main__":
    report = run_anomaly_detection()
    print(f"\n{' Pipeline Healthy' if report['pipeline_healthy'] else ' Anomalies Found'}")
    print(f"Checks: {report['total_checks']} | Anomalies: {report['anomalies_found']}")