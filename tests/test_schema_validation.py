"""
Unit tests for schema validation pipeline.
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from schema_validation import run_validation


@pytest.fixture
def valid_df():
    """Fixture providing a valid DataFrame matching expected schema."""
    size = 5000
    return pd.DataFrame({
        "system":                    ["SYSTEM: test"] * size,
        "chat":                      ["USER: hello ASSISTANT: hi"] * size,
        "num_turns":                 [1] * size,
        "num_calls":                 [1] * size,
        "complexity_tier":           ["simple"] * size,
        "has_parallel":              [False] * size,
        "has_malformed":             [False] * size,
        "function_calls":            ["[]"] * size,
        "num_defined_functions":     [1] * size,
        "defined_function_names":    ["[]"] * size,
        "function_signatures":       ["{}"] * size,
        "has_error_handling":        [False] * size,
        "has_function_error_response": [False] * size,
        "has_conditional_error":     [False] * size,
        "error_keywords_found":      ["[]"] * size,
    })


class TestRunValidation:

    def test_valid_df_passes_all(self, valid_df):
        """Test valid DataFrame passes all expectations."""
        results = run_validation(valid_df)
        assert all(results.values())

    def test_returns_dict(self, valid_df):
        """Test validation returns a dictionary."""
        results = run_validation(valid_df)
        assert isinstance(results, dict)

    def test_all_expected_keys_present(self, valid_df):
        """Test all expected expectation keys are in results."""
        results = run_validation(valid_df)
        assert "row_count" in results
        assert "complexity_tier_values" in results
        assert "no_nulls_chat" in results

    def test_invalid_complexity_fails(self, valid_df):
        """Test DataFrame with invalid complexity tier values fails."""
        valid_df["complexity_tier"] = "invalid_tier"
        results = run_validation(valid_df)
        assert results["complexity_tier_values"] is False