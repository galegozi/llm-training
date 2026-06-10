"""CPU-only preprocessing for the chess QLoRA dataset.

This stage intentionally runs before any Modal GPU function is started. It streams
raw examples from Hugging Face, renders the chat conversations with the base
model tokenizer's chat template, tokenizes/truncates them, and saves a finite
Hugging Face Dataset to disk for the GPU training stage to load directly.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, set_seed

from training.train_qlora import TrainDefaults, format_example


@dataclass(frozen=True)
class PreprocessDefaults:
    prepared_dataset_dir: str = "/vol/preprocessed/chess-llm/train"


def parse_args() -> argparse.Namespace:
    train_defaults = TrainDefaults()
    prep_defaults = PreprocessDefaults()
    parser = argparse.ArgumentParser(description="Preprocess and tokenize chess training data before GPU training starts.")
    parser.add_argument("--model-id", default=train_defaults.model_id)
    parser.add_argument("--dataset-id", default=train_defaults.dataset_id)
    parser.add_argument("--dataset-config", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--prepared-dataset-dir", default=prep_defaults.prepared_dataset_dir)
    parser.add_argument("--max-samples", type=int, default=500_000)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def iter_tokenized_examples(args: argparse.Namespace, tokenizer: AutoTokenizer) -> Iterator[dict[str, Any]]:
    dataset = load_dataset(
        args.dataset_id,
        name=args.dataset_config,
        split=args.split,
        streaming=True,
        token=os.environ.get("HF_TOKEN"),
    )
    if args.max_samples:
        dataset = dataset.take(args.max_samples)

    yielded = 0
    for example in dataset:
        text = format_example(example, tokenizer)
        if not text.strip():
            continue
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
            add_special_tokens=False,
        )
        if not tokenized["input_ids"]:
            continue
        yielded += 1
        input_ids = tokenized["input_ids"]
        yield {
            "input_ids": input_ids,
            "attention_mask": tokenized["attention_mask"],
            "labels": input_ids.copy(),
        }
    print(f"Prepared {yielded} tokenized examples.")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, token=os.environ.get("HF_TOKEN"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    output_path = Path(args.prepared_dataset_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_generator(lambda: iter_tokenized_examples(args, tokenizer))
    dataset.save_to_disk(str(output_path))
    tokenizer.save_pretrained(output_path / "tokenizer")
    print(f"Saved preprocessed dataset to {output_path}")


if __name__ == "__main__":
    main()
