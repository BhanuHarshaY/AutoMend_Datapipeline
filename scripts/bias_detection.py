"""
Bias Detection via Data Slicing for Glaive Function Calling v2
Analyzes dataset slices to detect representation bias and skew.
"""

import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

PROCESSED_FILE = Path(__file__).resolve().parents[1] / "data" / "processed" / "glaive_processed.jsonl"
BIAS_DIR       = Path(__file__).resolve().parents[1] / "data" / "processed" / "validation"

REPRESENTATION_THRESHOLD = 0.05


def load_data(filepath: Path) -> pd.DataFrame:
    """Load processed JSONL into DataFrame."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def add_slice_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer slicing features for bias analysis.
    These replace demographic features since Glaive has none.
    """
    df = df.copy()

    # Turn count bucket slice
    df["turn_bucket"] = pd.cut(
        df["num_turns"],
        bins=[0, 1, 3, 5, 100],
        labels=["single", "short", "medium", "long"],
        right=True
    )

    # Call count bucket slice
    df["call_bucket"] = pd.cut(
        df["num_calls"],
        bins=[-1, 0, 1, 2, 100],
        labels=["no_calls", "one_call", "two_calls", "many_calls"],
        right=True
    )

    # Has defined functions slice
    df["has_defined_functions"] = df["num_defined_functions"] > 0

    return df


def analyze_slice(df: pd.DataFrame, slice_col: str) -> pd.DataFrame:
    """
    Analyze representation and statistics for a given slice column.
    Returns a DataFrame with per slice statistics.
    """
    total = len(df)
    stats = []

    for slice_val, group in df.groupby(slice_col, observed=True):
        count      = len(group)
        proportion = count / total
        avg_turns  = group["num_turns"].mean()
        avg_calls  = group["num_calls"].mean()
        malformed  = group["has_malformed"].sum()
        error_pct  = group["has_error_handling"].mean()

        is_underrepresented = proportion < REPRESENTATION_THRESHOLD

        stats.append({
            "slice_column":        slice_col,
            "slice_value":         str(slice_val),
            "count":               count,
            "proportion":          round(proportion, 4),
            "avg_turns":           round(avg_turns, 2),
            "avg_calls":           round(avg_calls, 2),
            "malformed_count":     int(malformed),
            "error_handling_pct":  round(error_pct, 4),
            "is_underrepresented": is_underrepresented,
        })

    return pd.DataFrame(stats)


def detect_representation_bias(slice_df: pd.DataFrame) -> list:
    """
    Detect slices that are significantly underrepresented.
    Returns list of bias findings.
    """
    findings = []
    underrepresented = slice_df[slice_df["is_underrepresented"]]

    for _, row in underrepresented.iterrows():
        findings.append({
            "slice_column": row["slice_column"],
            "slice_value":  row["slice_value"],
            "proportion":   row["proportion"],
            "count":        row["count"],
            "severity":     "high" if row["proportion"] < 0.01 else "medium",
            "recommendation": (
                f"Slice '{row['slice_value']}' in '{row['slice_column']}' "
                f"represents only {row['proportion']:.2%} of data. "
                f"Consider oversampling or collecting more examples."
            )
        })

    return findings


def suggest_mitigation(findings: list) -> list:
    """
    Suggest mitigation strategies for detected bias.
    Per grading requirements bias mitigation must be documented.
    """
    mitigations = []
    for finding in findings:
        if finding["severity"] == "high":
            strategy = "oversample"
            detail   = (
                f"Apply SMOTE or random oversampling to '{finding['slice_value']}' "
                f"slice to bring representation above 5 percent threshold."
            )
        else:
            strategy = "collect_more"
            detail   = (
                f"Collect additional examples for '{finding['slice_value']}' "
                f"slice or apply sample weights during fine tuning."
            )

        mitigations.append({
            "slice_column": finding["slice_column"],
            "slice_value":  finding["slice_value"],
            "strategy":     strategy,
            "detail":       detail,
        })

    return mitigations


def print_bias_report(
    slice_results: dict,
    findings: list,
    mitigations: list
) -> None:
    """Print a clean bias detection report."""
    print("\n" + "=" * 60)
    print("          BIAS DETECTION REPORT")
    print("=" * 60)

    for slice_col, slice_df in slice_results.items():
        print(f"\nSlice: {slice_col}")
        print("-" * 40)
        print(slice_df[[
            "slice_value", "count", "proportion",
            "avg_turns", "avg_calls", "is_underrepresented"
        ]].to_string(index=False))

    print("\n" + "=" * 60)
    if findings:
        print(f"BIAS FINDINGS: {len(findings)} underrepresented slices detected")
        for f in findings:
            print(f"\n  Severity : {f['severity'].upper()}")
            print(f"  Slice    : {f['slice_column']} = {f['slice_value']}")
            print(f"  Coverage : {f['proportion']:.2%} ({f['count']} records)")
            print(f"  Action   : {f['recommendation']}")
    else:
        print("No significant bias detected across all slices.")

    if mitigations:
        print("\n" + "=" * 60)
        print("MITIGATION STRATEGIES:")
        for m in mitigations:
            print(f"\n  Slice    : {m['slice_column']} = {m['slice_value']}")
            print(f"  Strategy : {m['strategy']}")
            print(f"  Detail   : {m['detail']}")

    print("\n" + "=" * 60)


def run_bias_detection(filepath: Path = PROCESSED_FILE) -> dict:
    """Main bias detection pipeline."""
    BIAS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading processed data...")
    df = load_data(filepath)
    logger.info("Loaded %d records", len(df))

    logger.info("Engineering slice features...")
    df = add_slice_features(df)

    # Define slices to analyze
    slice_columns = [
        "complexity_tier",
        "turn_bucket",
        "call_bucket",
        "has_error_handling",
        "has_parallel",
        "has_defined_functions",
    ]

    # Analyze each slice
    slice_results = {}
    all_slice_dfs = []

    for col in slice_columns:
        logger.info("Analyzing slice: %s", col)
        slice_df = analyze_slice(df, col)
        slice_results[col] = slice_df
        all_slice_dfs.append(slice_df)

    # Combine all slice results
    combined_df = pd.concat(all_slice_dfs, ignore_index=True)

    # Detect bias
    logger.info("Detecting representation bias...")
    findings   = detect_representation_bias(combined_df)
    mitigations = suggest_mitigation(findings)

    logger.info("Found %d bias findings", len(findings))

    # Print report
    print_bias_report(slice_results, findings, mitigations)

    # Save report
    report = {
        "total_records":    len(df),
        "slices_analyzed":  len(slice_columns),
        "findings_count":   len(findings),
        "bias_detected":    len(findings) > 0,
        "findings":         findings,
        "mitigations":      mitigations,
        "slice_statistics": {
            col: slice_results[col].to_dict(orient="records")
            for col in slice_columns
        }
    }

    report_path = BIAS_DIR / "bias_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("Bias report saved to %s", report_path)
    return report


if __name__ == "__main__":
    report = run_bias_detection()
    print(f"\nBias detected: {report['bias_detected']}")
    print(f"Findings: {report['findings_count']}")
    print(f"Slices analyzed: {report['slices_analyzed']}")