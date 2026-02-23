"""
Preprocessing Script for Glaive Function Calling v2
Parses raw JSONL, extracts function calls, cleans and engineers features.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional
import pandas as pd

# Logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Config 
RAW_FILE      = Path(__file__).resolve().parents[1] / "data" / "raw" / "glaive_raw.jsonl"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
OUTPUT_FILE   = PROCESSED_DIR / "glaive_processed.jsonl"

# Error keywords for pattern detection
ERROR_KEYWORDS = [
    "error", "invalid", "failed", "null", "none",
    "exception", "traceback", "undefined", "not found"
]


#  Helpers 
def extract_function_signatures(system: str) -> dict:
    """
    Extract function names and parameter signatures from system prompt.
    Glaive format: plain text followed by a single JSON object.
    """
    signatures = {}

    if not system or not isinstance(system, str):
        return signatures

    try:
        # Glaive uses a single JSON object, not an array
        json_pattern = r"\{.*\}"
        match = re.search(json_pattern, system, re.DOTALL)
        if match:
            func = json.loads(match.group())
            if isinstance(func, dict) and "name" in func:
                name = func.get("name", "unknown")
                params = func.get("parameters", {})
                signatures[name] = {
                    "description": func.get("description", ""),
                    "parameters": list(
                        params.get("properties", {}).keys()
                    ) if isinstance(params, dict) else [],
                    "required": params.get("required", [])
                    if isinstance(params, dict) else [],
                }
    except (json.JSONDecodeError, AttributeError):
        pass

    return signatures


def extract_function_calls(text: str) -> list:
    """
    Extract JSON function call blocks from assistant text.
    Handles Glaive's format where arguments is a single-quoted string.
    """
    calls = []
    pattern = r"<functioncall>\s*(\{.*?\})\s*(?:<\|endoftext\|>|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            # Fix single quotes around arguments value
            fixed = re.sub(r"'(\{.*?\})'", r'\1', match, flags=re.DOTALL)
            parsed = json.loads(fixed)
            # Parse arguments if still a string
            if isinstance(parsed.get("arguments"), str):
                try:
                    parsed["arguments"] = json.loads(parsed["arguments"])
                except json.JSONDecodeError:
                    pass
            calls.append(parsed)
        except json.JSONDecodeError:
            calls.append({"__malformed__": match[:100]})
    return calls


def detect_error_handling(chat: str) -> dict:
    """
    Detect error handling patterns in the conversation.
    Checks if assistant responses contain error-related language
    or explicit error handling logic.
    Returns dict with detection results.
    """
    if not chat or not isinstance(chat, str):
        return {"has_error_handling": False, "error_keywords_found": []}

    chat_lower = chat.lower()

    # Find which error keywords appear
    found_keywords = [
        kw for kw in ERROR_KEYWORDS if kw in chat_lower
    ]

    # Check for explicit function error responses
    has_function_error = bool(
        re.search(r"<functionresponse>.*?(error|failed|invalid).*?</functionresponse>",
                  chat_lower, re.DOTALL)
    )

    # Check for conditional error handling language
    has_conditional_error = bool(
        re.search(r"(if|when).{0,30}(error|fail|invalid)", chat_lower)
    )

    return {
        "has_error_handling": len(found_keywords) > 0,
        "has_function_error_response": has_function_error,
        "has_conditional_error": has_conditional_error,
        "error_keywords_found": found_keywords,
    }


def count_turns(chat: str) -> int:
    """Count number of USER: turns in conversation."""
    return len(re.findall(r"USER:", chat))


def classify_complexity(calls: list) -> str:
    """
    Classify function call complexity for bias slicing.
    Adds 'refused' tier for when assistant declined to call any function.
    """
    if not calls:
        return "none"
    if len(calls) == 1:
        args = calls[0].get("arguments", {})
        if "__malformed__" in calls[0]:
            return "malformed"
        if isinstance(args, dict) and len(args) <= 2:
            return "simple"
        return "moderate"
    return "complex"


def has_malformed_calls(calls: list) -> bool:
    """Check if any extracted call is malformed."""
    return any("__malformed__" in c for c in calls)


def process_record(record: dict) -> Optional[dict]:
    """
    Process a single raw record into a structured format.
    Returns None if record is invalid.
    """
    system = record.get("system", "")
    chat   = record.get("chat", "")

    if not chat or not isinstance(chat, str):
        return None

    # Original features
    calls      = extract_function_calls(chat)
    turn_count = count_turns(chat)
    complexity = classify_complexity(calls)
    has_errors = has_malformed_calls(calls)
    has_parallel = len(calls) > 1

    # NEW: Extract function signatures from system prompt
    signatures     = extract_function_signatures(system)
    num_defined_fns = len(signatures)
    defined_fn_names = list(signatures.keys())

    # NEW: Detect error handling patterns
    error_info = detect_error_handling(chat)

    return {
        # Original fields
        "system":           system,
        "chat":             chat,
        "num_turns":        turn_count,
        "num_calls":        len(calls),
        "complexity_tier":  complexity,
        "has_parallel":     has_parallel,
        "has_malformed":    has_errors,
        "function_calls":   json.dumps(calls),

        # NEW: Function signature fields
        "num_defined_functions": num_defined_fns,
        "defined_function_names": json.dumps(defined_fn_names),
        "function_signatures":    json.dumps(signatures),

        # NEW: Error handling fields
        "has_error_handling":          error_info["has_error_handling"],
        "has_function_error_response": error_info["has_function_error_response"],
        "has_conditional_error":       error_info["has_conditional_error"],
        "error_keywords_found":        json.dumps(error_info["error_keywords_found"]),
    }

def remap_to_chatml(record: dict) -> dict:
    """
    Remap processed Glaive record to AutoMend ChatML format.
    
    Converts Glaive's function calling format into the standardized
    ChatML format required for Llama-3 fine-tuning.
    
    Format B (from scoping doc):
    {
        "messages": [
            {"role": "system", "content": "You are AutoMend. Available Tools: [...]"},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "{\"workflow\": {\"steps\": [...]}}"}
        ]
    }
    """
    system  = record.get("system", "")
    chat    = record.get("chat", "")
    calls   = json.loads(record.get("function_calls", "[]"))

    #  Extract user message──────────────────────────────────────────────
    # Get first USER: turn from chat
    user_match = re.search(r"USER:\s*(.*?)(?=ASSISTANT:|$)", chat, re.DOTALL)
    user_content = user_match.group(1).strip() if user_match else chat[:200]

    #  Build AutoMend system prompt
    # Keep original tool definitions from Glaive system prompt
    # This teaches model to respect provided context (per scoping doc)
    tool_definitions = record.get("function_signatures", "{}")
    system_content = (
        f"You are AutoMend, an MLOps remediation engine. "
        f"Convert user requests into valid JSON workflow definitions.\n"
        f"Available Tools: {tool_definitions}"
    )

    #  Build assistant response in AutoMend workflow format 
    if calls and not any("__malformed__" in c for c in calls):
        # Convert Glaive function calls to AutoMend workflow.steps format
        steps = []
        for call in calls:
            step = {
                "tool":   call.get("name", "unknown"),
                "params": call.get("arguments", {}),
            }
            steps.append(step)
        assistant_content = json.dumps({"workflow": {"steps": steps}}, indent=2)
    else:
        # No function call — assistant declined or no action needed
        assistant_match = re.search(
            r"ASSISTANT:\s*(.*?)(?=USER:|<\|endoftext\|>|$)",
            chat, re.DOTALL
        )
        raw_response = assistant_match.group(1).strip() if assistant_match else ""
        assistant_content = json.dumps({
            "workflow": {"steps": []},
            "message":  raw_response[:200]
        })

    return {
        "messages": [
            {"role": "system",    "content": system_content},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        # Keep metadata for filtering/analysis
        "complexity_tier":  record.get("complexity_tier"),
        "has_error_handling": record.get("has_error_handling"),
        "num_turns":        record.get("num_turns"),
        "num_calls":        record.get("num_calls"),
    }

def run_preprocessing(
    raw_file: Path = RAW_FILE,
    output_file: Path = OUTPUT_FILE,
) -> pd.DataFrame:
    """Main preprocessing pipeline."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading raw data from %s", raw_file)
    records = []
    with open(raw_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    logger.info("Loaded %d raw records", len(records))

    processed, skipped = [], 0
    for i, record in enumerate(records):
        result = process_record(record)
        if result is None:
            skipped += 1
        else:
            processed.append(result)
        if (i + 1) % 1000 == 0:
            logger.info("  Processed %d / %d records", i + 1, len(records))

    logger.info("Processing complete: %d valid, %d skipped", len(processed), skipped)

    df = pd.DataFrame(processed)

    # Log statistics
    logger.info("--- Dataset Statistics ---")
    logger.info("Total records:           %d", len(df))
    logger.info("Avg turns:               %.2f", df["num_turns"].mean())
    logger.info("Avg calls:               %.2f", df["num_calls"].mean())
    logger.info("Complexity breakdown:\n%s", df["complexity_tier"].value_counts().to_string())
    logger.info("Malformed calls:         %d", df["has_malformed"].sum())
    logger.info("Parallel calls:          %d", df["has_parallel"].sum())
    logger.info("Avg defined functions:   %.2f", df["num_defined_functions"].mean())
    logger.info("Records with error handling: %d", df["has_error_handling"].sum())
    logger.info("Records with fn errors:      %d", df["has_function_error_response"].sum())

    logger.info("Saving processed data to %s", output_file)
    with open(output_file, "w", encoding="utf-8") as f:
        for record in processed:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")



    logger.info("Preprocessing complete.")

    # ChatML Remapping
    # Remap all 5000 records to ChatML format for fine tuning
    logger.info("Remapping %d records to ChatML format...", len(processed))
    chatml_records = [remap_to_chatml(r) for r in processed]

    chatml_file = PROCESSED_DIR / "glaive_chatml.jsonl"
    with open(chatml_file, "w", encoding="utf-8") as f:
        for record in chatml_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("ChatML file saved: %d records to %s", len(chatml_records), chatml_file)

    return df


if __name__ == "__main__":
    df = run_preprocessing()
    print(f"\n Preprocessing complete. Shape: {df.shape}")
    print("\nComplexity breakdown:")
    print(df["complexity_tier"].value_counts())
    print(f"\nMalformed records:            {df['has_malformed'].sum()}")
    print(f"Records with error handling:  {df['has_error_handling'].sum()}")
    print(f"Avg defined functions:        {df['num_defined_functions'].mean():.2f}")
    # Print sample ChatML record for verification
    print("\nSample ChatML record:")
    with open(PROCESSED_DIR / "glaive_chatml.jsonl") as f:
        sample = json.loads(f.readline())
    for msg in sample["messages"]:
        print(f"\nRole: {msg['role']}")
        print(f"Content: {msg['content'][:200]}")