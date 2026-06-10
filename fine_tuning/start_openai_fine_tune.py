"""Launch an OpenAI supervised fine-tuning job from local JSONL training data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

SUPPORTED_METHODS = {"supervised", "dpo"}


class ConfigurationError(RuntimeError):
    """Raised when required local configuration is missing or invalid."""


def _json_object(line: str, line_number: int) -> dict[str, Any]:
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError(f"line {line_number}: expected a JSON object")
    return value


def validate_jsonl(path: Path, *, require_messages: bool) -> int:
    """Validate a JSONL file and return its example count."""
    if not path.exists():
        raise FileNotFoundError(f"training file not found: {path}")
    if not path.is_file():
        raise ValueError(f"training path is not a file: {path}")

    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                raise ValueError(f"line {line_number}: blank lines are not valid JSONL examples")
            example = _json_object(line, line_number)
            if require_messages:
                _validate_chat_example(example, line_number)
            count += 1

    if count == 0:
        raise ValueError(f"training file is empty: {path}")
    return count


def _validate_chat_example(example: dict[str, Any], line_number: int) -> None:
    messages = example.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"line {line_number}: expected a non-empty 'messages' array")

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"line {line_number}, message {index}: expected a JSON object")
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(
                f"line {line_number}, message {index}: invalid role {role!r}; "
                "expected system, user, assistant, or tool"
            )
        if "content" not in message and "tool_calls" not in message:
            raise ValueError(
                f"line {line_number}, message {index}: expected 'content' or 'tool_calls'"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upload JSONL data and create an OpenAI fine-tuning job."
    )
    parser.add_argument(
        "--training-file",
        type=Path,
        default=Path(os.environ.get("OPENAI_FINE_TUNE_TRAINING_FILE", "data/train.jsonl")),
        help="Path to JSONL training examples. Defaults to data/train.jsonl or "
        "$OPENAI_FINE_TUNE_TRAINING_FILE.",
    )
    parser.add_argument(
        "--validation-file",
        type=Path,
        default=(
            Path(os.environ["OPENAI_FINE_TUNE_VALIDATION_FILE"])
            if os.environ.get("OPENAI_FINE_TUNE_VALIDATION_FILE")
            else None
        ),
        help="Optional JSONL validation examples.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_FINE_TUNE_MODEL", "gpt-4o-mini-2024-07-18"),
        help="Base model to fine-tune. Defaults to $OPENAI_FINE_TUNE_MODEL or "
        "gpt-4o-mini-2024-07-18.",
    )
    parser.add_argument(
        "--method",
        choices=sorted(SUPPORTED_METHODS),
        default=os.environ.get("OPENAI_FINE_TUNE_METHOD", "supervised"),
        help="Fine-tuning method. Defaults to supervised.",
    )
    parser.add_argument(
        "--suffix",
        default=os.environ.get("OPENAI_FINE_TUNE_SUFFIX"),
        help="Optional model suffix for easier identification in the dashboard.",
    )
    parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Metadata pair to attach to the job. May be passed multiple times.",
    )
    parser.add_argument(
        "--skip-chat-validation",
        action="store_true",
        help="Only validate JSONL syntax, not chat-format 'messages'. Use for non-chat methods.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate local configuration and data without uploading files or creating a job.",
    )
    return parser


def parse_metadata(pairs: list[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"metadata must be KEY=VALUE, got {pair!r}")
        key, value = pair.split("=", 1)
        if not key:
            raise ValueError(f"metadata key cannot be empty in {pair!r}")
        metadata[key] = value
    return metadata


def create_fine_tuning_job(args: argparse.Namespace) -> dict[str, Any]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise ConfigurationError("OPENAI_API_KEY is not set; cannot call the OpenAI API")

    require_messages = not args.skip_chat_validation and args.method == "supervised"
    train_count = validate_jsonl(args.training_file, require_messages=require_messages)
    validation_count = None
    if args.validation_file:
        validation_count = validate_jsonl(args.validation_file, require_messages=require_messages)

    metadata = parse_metadata(args.metadata)
    if args.dry_run:
        return {
            "dry_run": True,
            "model": args.model,
            "method": args.method,
            "training_file": str(args.training_file),
            "training_examples": train_count,
            "validation_file": str(args.validation_file) if args.validation_file else None,
            "validation_examples": validation_count,
            "suffix": args.suffix,
            "metadata": metadata,
        }

    from openai import OpenAI

    client = OpenAI()
    with args.training_file.open("rb") as train_handle:
        uploaded_training_file = client.files.create(file=train_handle, purpose="fine-tune")

    uploaded_validation_file = None
    if args.validation_file:
        with args.validation_file.open("rb") as validation_handle:
            uploaded_validation_file = client.files.create(
                file=validation_handle, purpose="fine-tune"
            )

    job_args: dict[str, Any] = {
        "model": args.model,
        "training_file": uploaded_training_file.id,
        "method": {"type": args.method},
    }
    if uploaded_validation_file:
        job_args["validation_file"] = uploaded_validation_file.id
    if args.suffix:
        job_args["suffix"] = args.suffix
    if metadata:
        job_args["metadata"] = metadata

    job = client.fine_tuning.jobs.create(**job_args)
    return {
        "job_id": job.id,
        "status": job.status,
        "model": job.model,
        "training_file": uploaded_training_file.id,
        "validation_file": uploaded_validation_file.id if uploaded_validation_file else None,
        "fine_tuned_model": job.fine_tuned_model,
    }


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = create_fine_tuning_job(args)
    except (ConfigurationError, FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}")
        return 2

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
