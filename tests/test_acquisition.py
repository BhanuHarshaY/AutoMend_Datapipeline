"""
Unit tests for data acquisition pipeline.
"""

import json
import pytest
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from data_acquisition import fetch_and_save


class TestFetchAndSave:

    def test_output_file_created(self, tmp_path):
        """Test that output file is created after fetch."""
        output_file = tmp_path / "test_output.jsonl"
        count = fetch_and_save(
            sample_size=5,
            output_file=output_file
        )
        assert output_file.exists()
        assert count == 5

    def test_output_is_valid_jsonl(self, tmp_path):
        """Test that output file contains valid JSONL."""
        output_file = tmp_path / "test_output.jsonl"
        fetch_and_save(sample_size=5, output_file=output_file)
        with open(output_file) as f:
            for line in f:
                record = json.loads(line)
                assert isinstance(record, dict)

    def test_output_has_required_fields(self, tmp_path):
        """Test each record has system and chat fields."""
        output_file = tmp_path / "test_output.jsonl"
        fetch_and_save(sample_size=5, output_file=output_file)
        with open(output_file) as f:
            for line in f:
                record = json.loads(line)
                assert "system" in record
                assert "chat" in record

    def test_sample_size_respected(self, tmp_path):
        """Test that exactly sample_size records are saved."""
        output_file = tmp_path / "test_output.jsonl"
        count = fetch_and_save(sample_size=10, output_file=output_file)
        assert count == 10
        with open(output_file) as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) == 10