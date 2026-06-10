"""QLoRA SFT entrypoint for chess move prediction datasets on Hugging Face.

Designed for espressocheese/LichessGames rows shaped like:
{
  "whiteElo": int,
  "blackElo": int,
  "sysPlayer": "white" | "black",
  "conversations": [
    {"from": "system", "value": "..."},
    {"from": "assistant", "value": "..."}
  ]
}
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import IterableDataset, load_dataset, load_from_disk
from huggingface_hub import HfApi, create_repo, upload_file
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import SFTConfig, SFTTrainer

ROLE_MAP = {
    "system": "system",
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "bot": "assistant",
}


@dataclass(frozen=True)
class TrainDefaults:
    model_id: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    dataset_id: str = "espressocheese/LichessGames"
    output_dir: str = "/vol/checkpoints/chess-llm"
    hub_model_id: str = "espressocheese/chess-llm"


def parse_args() -> argparse.Namespace:
    defaults = TrainDefaults()
    parser = argparse.ArgumentParser(description="Fine-tune Nemotron Nano 30B-A3B on chess move data with QLoRA.")
    parser.add_argument("--model-id", default=defaults.model_id)
    parser.add_argument("--dataset-id", default=defaults.dataset_id)
    parser.add_argument("--dataset-config", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--hub-model-id", default=defaults.hub_model_id)
    parser.add_argument("--max-samples", type=int, default=500_000, help="Cap streamed examples for budget control.")
    parser.add_argument("--max-steps", type=int, default=4_000, help="Hard training-step cap for Modal cost control.")
    parser.add_argument("--prepared-dataset-dir", default="", help="Path to a CPU-preprocessed/tokenized dataset saved with datasets.save_to_disk.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--no-streaming", action="store_true", help="Disable streaming. Not recommended for this dataset.")
    parser.add_argument("--merge-and-upload", action="store_true", help="Merge LoRA into the base model before upload. Requires much more VRAM/RAM.")
    parser.add_argument("--private", action="store_true", help="Upload the Hugging Face repo as private. Default is public.")
    parser.add_argument("--report-to", default="none", choices=["none", "wandb", "tensorboard"])
    return parser.parse_args()


def normalize_messages(example: dict[str, Any]) -> list[dict[str, str]]:
    conversations = example.get("conversations") or []
    messages: list[dict[str, str]] = []
    for message in conversations:
        role = ROLE_MAP.get(str(message.get("from", "")).lower())
        content = message.get("value")
        if role and content:
            messages.append({"role": role, "content": str(content)})
    return messages


def format_example(example: dict[str, Any], tokenizer: AutoTokenizer) -> str:
    messages = normalize_messages(example)
    if not messages:
        return ""
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    # Fallback for tokenizers without a chat template.
    rendered: list[str] = []
    for message in messages:
        rendered.append(f"<{message['role']}>\n{message['content']}")
    return "\n".join(rendered) + tokenizer.eos_token


def build_dataset(args: argparse.Namespace) -> IterableDataset:
    if args.prepared_dataset_dir:
        return load_from_disk(args.prepared_dataset_dir)

    dataset = load_dataset(
        args.dataset_id,
        name=args.dataset_config,
        split=args.split,
        streaming=not args.no_streaming,
        token=os.environ.get("HF_TOKEN"),
    )
    if args.max_samples and not args.no_streaming:
        dataset = dataset.take(args.max_samples)
    elif args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    return dataset


def write_model_card(args: argparse.Namespace, output_dir: str) -> Path:
    visibility = "private" if args.private else "public"
    card = f"""---
base_model: {args.model_id}
datasets:
- {args.dataset_id}
library_name: peft
tags:
- chess
- qlora
- modal
- nemotron
---

# chess-llm

This repository contains a LoRA/QLoRA fine-tune of `{args.model_id}` on `{args.dataset_id}` for chess move-selection prompts.

- Training launcher: Modal
- Upload visibility: {visibility}
- Max training steps for this run: {args.max_steps}
- Max streamed samples for this run: {args.max_samples}
- Preprocessed dataset directory: {args.prepared_dataset_dir or "not used"}
- Max sequence length: {args.max_seq_length}
- LoRA rank/alpha/dropout: {args.lora_r}/{args.lora_alpha}/{args.lora_dropout}

The default artifact is a PEFT adapter. Load it with the base model named above unless the run was launched with `--merge-and-upload`.
"""
    path = Path(output_dir) / "README.md"
    path.write_text(card, encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, token=os.environ.get("HF_TOKEN"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    train_dataset = build_dataset(args)
    using_preprocessed_dataset = bool(args.prepared_dataset_dir)

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        max_length=args.max_seq_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        max_steps=args.max_steps,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        optim="paged_adamw_8bit",
        report_to=[] if args.report_to == "none" else [args.report_to],
        push_to_hub=False,
        remove_unused_columns=False,
    )

    def formatting_prompts_func(examples: Any) -> str | list[str]:
        if isinstance(examples, list):
            return [format_example(item, tokenizer) for item in examples]
        if isinstance(examples, dict) and examples:
            first_value = next(iter(examples.values()))
            if isinstance(first_value, list):
                rows = [dict(zip(examples.keys(), values)) for values in zip(*examples.values())]
                return [format_example(row, tokenizer) for row in rows]
        return format_example(examples, tokenizer)

    trainer_kwargs = {
        "model": model,
        "args": sft_config,
        "train_dataset": train_dataset,
        "peft_config": lora_config,
        "processing_class": tokenizer,
    }
    if not using_preprocessed_dataset:
        trainer_kwargs["formatting_func"] = formatting_prompts_func

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    private = bool(args.private)
    token = os.environ.get("HF_TOKEN")
    create_repo(args.hub_model_id, private=private, exist_ok=True, token=token)
    HfApi(token=token).update_repo_visibility(repo_id=args.hub_model_id, private=private, repo_type="model")
    model_card_path = write_model_card(args, args.output_dir)
    if args.merge_and_upload:
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(args.output_dir, safe_serialization=True, max_shard_size="5GB")
        tokenizer.save_pretrained(args.output_dir)
        merged.push_to_hub(args.hub_model_id, private=private, token=token, max_shard_size="5GB")
    else:
        trainer.model.push_to_hub(args.hub_model_id, private=private, token=token)
    tokenizer.push_to_hub(args.hub_model_id, private=private, token=token)
    upload_file(
        path_or_fileobj=str(model_card_path),
        path_in_repo="README.md",
        repo_id=args.hub_model_id,
        repo_type="model",
        token=token,
    )


if __name__ == "__main__":
    main()
