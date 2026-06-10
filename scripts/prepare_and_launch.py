"""Prepare Hugging Face/Modal auth, make the target repo public, and launch training.

This script intentionally never prints token values. It uses your locally configured
Hugging Face token and Modal credentials, then creates/updates the Modal secret that
the remote training job needs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import subprocess
import sys
from pathlib import Path


DEFAULT_HUB_MODEL_ID = "espressocheese/chess-llm"
DEFAULT_MAX_STEPS = 4_000
DEFAULT_MAX_SAMPLES = 500_000


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


def require_executable(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(
            f"Missing required executable: {name}. "
            "Install local launcher dependencies with: "
            "python -m pip install -r requirements-launch.txt"
        )


def load_dotenv(path: Path = Path(".env")) -> None:
    """Load KEY=VALUE pairs from .env if present, without overriding env vars."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_hf_token() -> str:
    """Resolve a Hugging Face token from env first, then local HF login."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "No HF_TOKEN/HUGGING_FACE_HUB_TOKEN env var was found, and "
            "huggingface_hub is not installed to read a local login. Install "
            "local launcher dependencies with: python -m pip install -r "
            "requirements-launch.txt"
        ) from exc
    token = get_token()
    if not token:
        raise SystemExit(
            "No Hugging Face token found. Set HF_TOKEN or run "
            "`huggingface-cli login` once, then re-run this script."
        )
    return token


def ensure_public_hf_repo(hub_model_id: str, hf_token: str) -> None:
    try:
        from huggingface_hub import HfApi
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing Python package: huggingface_hub. Install local launcher "
            "dependencies with: python -m pip install -r requirements-launch.txt"
        ) from exc
    api = HfApi(token=hf_token)
    api.create_repo(repo_id=hub_model_id, repo_type="model", private=False, exist_ok=True)
    api.update_repo_visibility(repo_id=hub_model_id, repo_type="model", private=False)
    print(f"Hugging Face model repo is public: https://huggingface.co/{hub_model_id}")


def upsert_modal_hf_secret(hf_token: str) -> None:
    require_executable("modal")
    with tempfile.NamedTemporaryFile("w", prefix="hf-token-", suffix=".json", delete=False) as handle:
        json.dump({"HF_TOKEN": hf_token}, handle)
        secret_path = handle.name
    os.chmod(secret_path, 0o600)
    try:
        run(["modal", "secret", "create", "--from-json", secret_path, "--force", "huggingface"])
    finally:
        os.remove(secret_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-command setup and launch for the chess LLM Modal run.")
    parser.add_argument("--hub-model-id", default=DEFAULT_HUB_MODEL_ID)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--gpu", default="H100", help="Modal GPU type passed via MODAL_GPU. Defaults to H100.")
    parser.add_argument("--report-to", default="none", choices=["none", "wandb", "tensorboard"])
    parser.add_argument("--merge-and-upload", action="store_true")
    parser.add_argument("--skip-launch", action="store_true", help="Only set up auth/repo/secret; do not start Modal training.")
    parser.add_argument("--skip-preprocessing", action="store_true", help="Reuse an existing preprocessed Modal volume dataset instead of rebuilding it before GPU training.")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    require_executable("modal")

    hf_token = resolve_hf_token()

    run(["modal", "token", "current"])
    ensure_public_hf_repo(args.hub_model_id, hf_token)
    upsert_modal_hf_secret(hf_token)

    if args.skip_launch:
        return

    env = os.environ.copy()
    env["MODAL_GPU"] = args.gpu
    cmd = [
        "modal",
        "run",
        "modal_train.py",
        "--max-steps",
        str(args.max_steps),
        "--max-samples",
        str(args.max_samples),
        "--learning-rate",
        str(args.learning_rate),
        "--hub-model-id",
        args.hub_model_id,
        "--max-seq-length",
        str(args.max_seq_length),
        "--report-to",
        args.report_to,
    ]
    if args.merge_and_upload:
        cmd.append("--merge-and-upload")
    if args.skip_preprocessing:
        cmd.append("--skip-preprocessing")
    run(cmd, env=env)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
