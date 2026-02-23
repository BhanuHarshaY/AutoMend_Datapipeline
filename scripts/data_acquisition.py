"""
Data Acquisition Script for Glaive Function Calling v2
Streams dataset from HuggingFace and saves a reproducible sample.
"""

import json
import logging
import os
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

#  Config 
load_dotenv()

DATASET_NAME = "glaiveai/glaive-function-calling-v2"
SAMPLE_SIZE  = 5000
RANDOM_SEED  = 42
RAW_DIR      = Path(__file__).resolve().parents[1] / "data" / "raw"
OUTPUT_FILE  = RAW_DIR / "glaive_raw.jsonl"


def fetch_and_save(
    dataset_name: str = DATASET_NAME,
    sample_size: int = SAMPLE_SIZE,
    output_file: Path = OUTPUT_FILE,
) -> int:
    """
    Stream dataset from HuggingFace, sample, and save as JSONL.
    Returns number of records saved.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Starting streaming from HuggingFace: %s", dataset_name)

    try:
        dataset = load_dataset(
            dataset_name,
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:
        logger.error("Failed to load dataset: %s", e)
        raise

    logger.info("Collecting %d samples...", sample_size)
    records = []
    for i, record in enumerate(dataset):
        if i >= sample_size:
            break
        records.append(record)
        if (i + 1) % 500 == 0:
            logger.info("  Collected %d / %d records", i + 1, sample_size)

    logger.info("Saving %d records to %s", len(records), output_file)
    with open(output_file, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Data acquisition complete. Records saved: %d", len(records))
    return len(records)


if __name__ == "__main__":
    count = fetch_and_save()
    print(f"\n Successfully saved {count} records to {OUTPUT_FILE}")