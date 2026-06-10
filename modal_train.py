"""Modal launcher for QLoRA fine-tuning and upload to Hugging Face.

Usage:
  modal secret create huggingface HF_TOKEN=hf_...
  modal run modal_train.py --max-steps 4000 --max-samples 500000

The local entrypoint runs CPU preprocessing first, then starts the GPU training
function only after the tokenized dataset has been saved to the shared volume.
"""

from __future__ import annotations

import os
import subprocess

import modal

GPU_TYPE = os.environ.get("MODAL_GPU", "H100")
PREPARED_DATASET_DIR = "/vol/preprocessed/chess-llm/train"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install_from_requirements("requirements.txt")
)

app = modal.App("chess-llm-finetune")
volume = modal.Volume.from_name("chess-llm-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    timeout=60 * 60 * 12,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/vol": volume},
)
def preprocess_dataset(
    max_samples: int = 500_000,
    max_seq_length: int = 1024,
    prepared_dataset_dir: str = PREPARED_DATASET_DIR,
) -> None:
    cmd = [
        "python",
        "training/preprocess_dataset.py",
        "--max-samples",
        str(max_samples),
        "--max-seq-length",
        str(max_seq_length),
        "--prepared-dataset-dir",
        prepared_dataset_dir,
    ]
    subprocess.run(cmd, check=True)
    volume.commit()


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=60 * 60 * 24,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/vol": volume},
)
def train(
    max_steps: int = 4_000,
    max_samples: int = 500_000,
    learning_rate: float = 2e-4,
    max_seq_length: int = 1024,
    hub_model_id: str = "espressocheese/chess-llm",
    report_to: str = "none",
    merge_and_upload: bool = False,
    private: bool = False,
    prepared_dataset_dir: str = PREPARED_DATASET_DIR,
) -> None:
    cmd = [
        "python",
        "training/train_qlora.py",
        "--max-steps",
        str(max_steps),
        "--max-samples",
        str(max_samples),
        "--learning-rate",
        str(learning_rate),
        "--max-seq-length",
        str(max_seq_length),
        "--hub-model-id",
        hub_model_id,
        "--report-to",
        report_to,
        "--prepared-dataset-dir",
        prepared_dataset_dir,
    ]
    if merge_and_upload:
        cmd.append("--merge-and-upload")
    if private:
        cmd.append("--private")
    subprocess.run(cmd, check=True)
    volume.commit()


@app.local_entrypoint()
def main(
    max_steps: int = 4_000,
    max_samples: int = 500_000,
    learning_rate: float = 2e-4,
    max_seq_length: int = 1024,
    hub_model_id: str = "espressocheese/chess-llm",
    report_to: str = "none",
    merge_and_upload: bool = False,
    private: bool = False,
    prepared_dataset_dir: str = PREPARED_DATASET_DIR,
    skip_preprocessing: bool = False,
) -> None:
    if not skip_preprocessing:
        preprocess_dataset.remote(
            max_samples=max_samples,
            max_seq_length=max_seq_length,
            prepared_dataset_dir=prepared_dataset_dir,
        )

    train.remote(
        max_steps=max_steps,
        max_samples=max_samples,
        learning_rate=learning_rate,
        max_seq_length=max_seq_length,
        hub_model_id=hub_model_id,
        report_to=report_to,
        merge_and_upload=merge_and_upload,
        private=private,
        prepared_dataset_dir=prepared_dataset_dir,
    )
