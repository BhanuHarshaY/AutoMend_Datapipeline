"""
Schema Validation for Glaive Function Calling v2
Uses Great Expectations to validate processed data quality.
"""

import json
import logging
from pathlib import Path

import pandas as pd
import great_expectations as ge

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Config
PROCESSED_FILE = Path(__file__).resolve().parents[1] / "data" / "processed" / "glaive_processed.jsonl"
VALIDATION_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "validation"


def load_processed_data(filepath: Path) -> pd.DataFrame:
    """Load processed JSONL into a DataFrame."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def run_validation(df: pd.DataFrame) -> dict:
    """
    Run Great Expectations validation suite on processed DataFrame.
    Returns validation results summary.
    """
    logger.info("Converting DataFrame to GE dataset...")
    ge_df = ge.from_pandas(df)

    results = {}

    # 1. Required columns exist
    logger.info("Validating required columns...")
    required_columns = [
        "system", "chat", "num_turns", "num_calls",
        "complexity_tier", "has_parallel", "has_malformed",
        "function_calls", "num_defined_functions",
        "defined_function_names", "function_signatures",
        "has_error_handling", "has_function_error_response",
        "has_conditional_error", "error_keywords_found"
    ]
    for col in required_columns:
        result = ge_df.expect_column_to_exist(col)
        results[f"column_exists_{col}"] = result["success"]

    #  2. No nulls in critical columns 
    logger.info("Validating no nulls in critical columns...")
    critical_cols = ["chat", "num_turns", "num_calls", "complexity_tier"]
    for col in critical_cols:
        result = ge_df.expect_column_values_to_not_be_null(col)
        results[f"no_nulls_{col}"] = result["success"]

    # 3. Numeric column ranges 
    logger.info("Validating numeric ranges...")

    result = ge_df.expect_column_values_to_be_between(
        "num_turns", min_value=0, max_value=50
    )
    results["num_turns_range"] = result["success"]

    result = ge_df.expect_column_values_to_be_between(
        "num_calls", min_value=0, max_value=20
    )
    results["num_calls_range"] = result["success"]

    result = ge_df.expect_column_values_to_be_between(
        "num_defined_functions", min_value=0, max_value=20
    )
    results["num_defined_functions_range"] = result["success"]

    #  4. Complexity tier values 
    logger.info("Validating complexity tier values...")
    result = ge_df.expect_column_values_to_be_in_set(
        "complexity_tier",
        ["none", "simple", "moderate", "complex", "malformed"]
    )
    results["complexity_tier_values"] = result["success"]

    #  5. Boolean columns 
    logger.info("Validating boolean columns...")
    bool_cols = [
        "has_parallel", "has_malformed",
        "has_error_handling", "has_function_error_response",
        "has_conditional_error"
    ]
    for col in bool_cols:
        result = ge_df.expect_column_values_to_be_in_set(
            col, [True, False]
        )
        results[f"bool_values_{col}"] = result["success"]

    #  6. Chat field not empty 
    logger.info("Validating chat field not empty...")
    result = ge_df.expect_column_value_lengths_to_be_between(
        "chat", min_value=10
    )
    results["chat_not_empty"] = result["success"]

    #  7. Dataset size 
    logger.info("Validating dataset size...")
    result = ge_df.expect_table_row_count_to_be_between(
        min_value=4000, max_value=6000
    )
    results["row_count"] = result["success"]

    return results


def print_validation_report(results: dict) -> None:
    """Print a clean validation report."""
    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    total  = len(results)

    print("\n" + "="*55)
    print("       GREAT EXPECTATIONS VALIDATION REPORT")
    print("="*55)

    # Print failed ones first
    failures = {k: v for k, v in results.items() if not v}
    if failures:
        print("\n FAILED EXPECTATIONS:")
        for name in failures:
            print(f"   - {name}")

    print(f"\n Passed: {passed}/{total}")
    print(f" Failed: {failed}/{total}")
    print(f" Success Rate: {passed/total*100:.1f}%")
    print("="*55)


def save_validation_report(results: dict) -> None:
    """Save validation results to JSON file."""
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    report_path = VALIDATION_DIR / "validation_report.json"

    report = {
        "total_expectations": len(results),
        "passed": sum(1 for v in results.values() if v),
        "failed": sum(1 for v in results.values() if not v),
        "success_rate": sum(1 for v in results.values() if v) / len(results),
        "results": results
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Validation report saved to %s", report_path)


if __name__ == "__main__":
    logger.info("Loading processed data...")
    df = load_processed_data(PROCESSED_FILE)
    logger.info("Loaded %d records", len(df))

    results = run_validation(df)
    print_validation_report(results)
    save_validation_report(results)