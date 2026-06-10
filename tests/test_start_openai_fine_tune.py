from __future__ import annotations

import json
from pathlib import Path

import pytest

from fine_tuning.start_openai_fine_tune import parse_metadata, validate_jsonl


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_validate_chat_jsonl_counts_examples(tmp_path: Path) -> None:
    training_file = tmp_path / "train.jsonl"
    write_jsonl(
        training_file,
        [
            {
                "messages": [
                    {"role": "system", "content": "Answer concisely."},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi!"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "2+2?"},
                    {"role": "assistant", "content": "4"},
                ]
            },
        ],
    )

    assert validate_jsonl(training_file, require_messages=True) == 2


def test_validate_chat_jsonl_rejects_missing_messages(tmp_path: Path) -> None:
    training_file = tmp_path / "train.jsonl"
    write_jsonl(training_file, [{"prompt": "Hello", "completion": "Hi"}])

    with pytest.raises(ValueError, match="messages"):
        validate_jsonl(training_file, require_messages=True)


def test_parse_metadata() -> None:
    assert parse_metadata(["project=llm-training", "owner=codex"]) == {
        "project": "llm-training",
        "owner": "codex",
    }


def test_parse_metadata_rejects_missing_equals() -> None:
    with pytest.raises(ValueError, match="KEY=VALUE"):
        parse_metadata(["project"])
