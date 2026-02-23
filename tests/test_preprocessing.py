"""
Unit tests for preprocessing pipeline.
Tests parsing, feature extraction, and edge cases.
"""

import json
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from preprocessing import (
    extract_function_calls,
    extract_function_signatures,
    detect_error_handling,
    count_turns,
    classify_complexity,
    has_malformed_calls,
    remap_to_chatml,
    process_record,
)


class TestExtractFunctionCalls:

    def test_valid_function_call(self):
        """Test extraction of a valid Glaive format function call."""
        chat = "ASSISTANT: <functioncall> {\"name\": \"get_weather\", \"arguments\": '{\"city\": \"Boston\"}'} <|endoftext|>"
        calls = extract_function_calls(chat)
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"

    def test_no_function_call(self):
        """Test plain assistant response returns empty list."""
        chat = "ASSISTANT: I cannot help with that. <|endoftext|>"
        calls = extract_function_calls(chat)
        assert calls == []

    def test_multiple_function_calls(self):
        """Test extraction of parallel function calls."""
        chat = (
            "ASSISTANT: <functioncall> {\"name\": \"func1\", \"arguments\": '{\"a\": \"1\"}'} <|endoftext|> "
            "ASSISTANT: <functioncall> {\"name\": \"func2\", \"arguments\": '{\"b\": \"2\"}'} <|endoftext|>"
        )
        calls = extract_function_calls(chat)
        assert len(calls) == 2

    def test_malformed_json_flagged(self):
        """Test malformed JSON is flagged not silently dropped."""
        chat = "ASSISTANT: <functioncall> {\"name\": \"func\", \"arguments\": '{broken json'} <|endoftext|>"
        calls = extract_function_calls(chat)
        assert any("__malformed__" in c for c in calls)

    def test_empty_chat(self):
        """Test empty string returns empty list."""
        calls = extract_function_calls("")
        assert calls == []

    def test_arguments_parsed_as_dict(self):
        """Test that arguments string is parsed into a dict."""
        chat = "ASSISTANT: <functioncall> {\"name\": \"get_weather\", \"arguments\": '{\"city\": \"Boston\"}'} <|endoftext|>"
        calls = extract_function_calls(chat)
        assert isinstance(calls[0]["arguments"], dict)
        assert calls[0]["arguments"]["city"] == "Boston"


class TestExtractFunctionSignatures:

    def test_valid_system_prompt(self):
        """Test extraction from a valid Glaive system prompt."""
        system = """SYSTEM: You are a helpful assistant.
        {
            "name": "get_weather",
            "description": "Get weather info",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "country": {"type": "string"}
                },
                "required": ["city"]
            }
        }"""
        sigs = extract_function_signatures(system)
        assert "get_weather" in sigs
        assert "city" in sigs["get_weather"]["parameters"]
        assert "city" in sigs["get_weather"]["required"]

    def test_empty_system(self):
        """Test empty system prompt returns empty dict."""
        sigs = extract_function_signatures("")
        assert sigs == {}

    def test_no_json_in_system(self):
        """Test system prompt with no JSON returns empty dict."""
        sigs = extract_function_signatures("You are a helpful assistant.")
        assert sigs == {}

    def test_none_system(self):
        """Test None system prompt handled gracefully."""
        sigs = extract_function_signatures(None)
        assert sigs == {}

    def test_signature_has_required_keys(self):
        """Test extracted signature contains description, parameters, required."""
        system = """SYSTEM: You are helpful.
        {
            "name": "test_func",
            "description": "A test function",
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"]
            }
        }"""
        sigs = extract_function_signatures(system)
        assert "description" in sigs["test_func"]
        assert "parameters" in sigs["test_func"]
        assert "required" in sigs["test_func"]


class TestDetectErrorHandling:

    def test_detects_error_keyword(self):
        """Test detection of error keyword in chat."""
        chat = "ASSISTANT: There was an error processing your request."
        result = detect_error_handling(chat)
        assert result["has_error_handling"] == True
        assert "error" in result["error_keywords_found"]

    def test_detects_invalid_keyword(self):
        """Test detection of invalid keyword."""
        chat = "ASSISTANT: Invalid parameter provided."
        result = detect_error_handling(chat)
        assert result["has_error_handling"] == True

    def test_no_error_keywords(self):
        """Test clean chat has no error handling detected."""
        chat = "ASSISTANT: Here is the weather in Boston today."
        result = detect_error_handling(chat)
        assert result["has_error_handling"] == False
        assert result["error_keywords_found"] == []

    def test_empty_chat(self):
        """Test empty chat handled gracefully."""
        result = detect_error_handling("")
        assert result["has_error_handling"] == False

    def test_multiple_keywords(self):
        """Test multiple error keywords are all detected."""
        chat = "ASSISTANT: Invalid request. Function returned null."
        result = detect_error_handling(chat)
        assert result["has_error_handling"] == True
        assert len(result["error_keywords_found"]) >= 2

    def test_function_error_response_detected(self):
        """Test explicit function error response detected."""
        chat = "ASSISTANT: <functionresponse> error occurred </functionresponse>"
        result = detect_error_handling(chat)
        assert result["has_function_error_response"] == True

    def test_result_has_all_keys(self):
        """Test result contains all expected keys."""
        result = detect_error_handling("test")
        assert "has_error_handling" in result
        assert "has_function_error_response" in result
        assert "has_conditional_error" in result
        assert "error_keywords_found" in result


class TestCountTurns:

    def test_single_turn(self):
        """Test single USER turn counted correctly."""
        chat = "USER: Hello ASSISTANT: Hi"
        assert count_turns(chat) == 1

    def test_multiple_turns(self):
        """Test multiple USER turns counted correctly."""
        chat = "USER: Hi ASSISTANT: Hello USER: Thanks ASSISTANT: Welcome"
        assert count_turns(chat) == 2

    def test_no_turns(self):
        """Test chat with no USER turns returns zero."""
        chat = "ASSISTANT: Hello there"
        assert count_turns(chat) == 0

    def test_empty_string(self):
        """Test empty string returns zero."""
        assert count_turns("") == 0


class TestClassifyComplexity:

    def test_none_complexity(self):
        """Test empty calls classified as none."""
        assert classify_complexity([]) == "none"

    def test_simple_complexity(self):
        """Test single call with few args classified as simple."""
        calls = [{"name": "get_weather", "arguments": {"city": "Boston"}}]
        assert classify_complexity(calls) == "simple"

    def test_moderate_complexity(self):
        """Test single call with many args classified as moderate."""
        calls = [{"name": "func", "arguments": {"a": 1, "b": 2, "c": 3}}]
        assert classify_complexity(calls) == "moderate"

    def test_complex_complexity(self):
        """Test multiple calls classified as complex."""
        calls = [
            {"name": "func1", "arguments": {}},
            {"name": "func2", "arguments": {}},
        ]
        assert classify_complexity(calls) == "complex"

    def test_malformed_complexity(self):
        """Test malformed call classified as malformed."""
        calls = [{"__malformed__": "broken"}]
        assert classify_complexity(calls) == "malformed"


class TestHasMalformedCalls:

    def test_no_malformed(self):
        """Test clean calls returns False."""
        calls = [{"name": "func", "arguments": {}}]
        assert has_malformed_calls(calls) == False

    def test_has_malformed(self):
        """Test malformed call returns True."""
        calls = [{"__malformed__": "broken json"}]
        assert has_malformed_calls(calls) == True

    def test_mixed_calls(self):
        """Test mixed valid and malformed returns True."""
        calls = [
            {"name": "func", "arguments": {}},
            {"__malformed__": "broken"},
        ]
        assert has_malformed_calls(calls) == True

    def test_empty_calls(self):
        """Test empty list returns False."""
        assert has_malformed_calls([]) == False


class TestProcessRecord:

    def test_valid_record(self):
        """Test valid record returns all required fields."""
        record = {
            "system": "SYSTEM: You are helpful. {\"name\": \"test_func\", \"description\": \"test\", \"parameters\": {\"type\": \"object\", \"properties\": {\"x\": {\"type\": \"string\"}}, \"required\": [\"x\"]}}",
            "chat": "USER: Help me ASSISTANT: Sure <|endoftext|>"
        }
        result = process_record(record)
        assert result is not None
        assert "complexity_tier" in result
        assert "num_turns" in result
        assert "function_calls" in result
        assert "has_error_handling" in result
        assert "num_defined_functions" in result

    def test_invalid_record_returns_none(self):
        """Test record with empty chat returns None."""
        record = {"system": "test", "chat": ""}
        result = process_record(record)
        assert result is None

    def test_missing_chat_returns_none(self):
        """Test record with missing chat field returns None."""
        record = {"system": "test"}
        result = process_record(record)
        assert result is None

    def test_record_with_function_call(self):
        """Test record containing function call is processed correctly."""
        record = {
            "system": "SYSTEM: You are helpful.",
            "chat": "USER: Get weather ASSISTANT: <functioncall> {\"name\": \"get_weather\", \"arguments\": '{\"city\": \"Boston\"}'} <|endoftext|>"
        }
        result = process_record(record)
        assert result is not None
        assert result["num_calls"] == 1
        assert result["complexity_tier"] in ["simple", "moderate"]

    def test_record_num_turns_correct(self):
        """Test turn count is correctly assigned."""
        record = {
            "system": "SYSTEM: test",
            "chat": "USER: Hi ASSISTANT: Hello USER: Thanks ASSISTANT: Welcome"
        }
        result = process_record(record)
        assert result["num_turns"] == 2


class TestRemapToChatML:

    def test_chatml_structure(self):
        """Test ChatML output has correct message structure."""
        record = {
            "system": "SYSTEM: You are helpful.",
            "chat": "USER: Help me ASSISTANT: Sure",
            "function_calls": "[]",
            "function_signatures": "{}",
            "complexity_tier": "none",
            "has_error_handling": False,
            "num_turns": 1,
            "num_calls": 0,
        }
        result = remap_to_chatml(record)
        assert "messages" in result
        assert len(result["messages"]) == 3
        roles = [m["role"] for m in result["messages"]]
        assert roles == ["system", "user", "assistant"]

    def test_chatml_assistant_is_valid_json(self):
        """Test assistant content in ChatML is valid JSON."""
        record = {
            "system": "SYSTEM: You are helpful.",
            "chat": "USER: Help me ASSISTANT: Sure",
            "function_calls": "[]",
            "function_signatures": "{}",
            "complexity_tier": "none",
            "has_error_handling": False,
            "num_turns": 1,
            "num_calls": 0,
        }
        result = remap_to_chatml(record)
        assistant_content = result["messages"][2]["content"]
        parsed = json.loads(assistant_content)
        assert "workflow" in parsed
        assert "steps" in parsed["workflow"]

    def test_chatml_system_contains_automend(self):
        """Test system message contains AutoMend context."""
        record = {
            "system": "SYSTEM: You are helpful.",
            "chat": "USER: Help me ASSISTANT: Sure",
            "function_calls": "[]",
            "function_signatures": "{}",
            "complexity_tier": "none",
            "has_error_handling": False,
            "num_turns": 1,
            "num_calls": 0,
        }
        result = remap_to_chatml(record)
        assert "AutoMend" in result["messages"][0]["content"]

    def test_chatml_has_metadata(self):
        """Test ChatML output retains metadata fields."""
        record = {
            "system": "SYSTEM: test",
            "chat": "USER: test ASSISTANT: test",
            "function_calls": "[]",
            "function_signatures": "{}",
            "complexity_tier": "simple",
            "has_error_handling": False,
            "num_turns": 1,
            "num_calls": 0,
        }
        result = remap_to_chatml(record)
        assert "complexity_tier" in result
        assert "num_turns" in result
        assert "num_calls" in result

    def test_chatml_with_function_calls(self):
        """Test ChatML correctly maps function calls to workflow steps."""
        record = {
            "system": "SYSTEM: test",
            "chat": "USER: Get weather ASSISTANT: done",
            "function_calls": "[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Boston\"}}]",
            "function_signatures": "{}",
            "complexity_tier": "simple",
            "has_error_handling": False,
            "num_turns": 1,
            "num_calls": 1,
        }
        result = remap_to_chatml(record)
        assistant_content = json.loads(result["messages"][2]["content"])
        assert len(assistant_content["workflow"]["steps"]) == 1
        assert assistant_content["workflow"]["steps"][0]["tool"] == "get_weather"