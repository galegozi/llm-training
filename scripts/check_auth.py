"""Check local Hugging Face and Modal authentication without printing secrets."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


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


def run(cmd: list[str]) -> int:
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=False).returncode


def env_status(name: str) -> None:
    value = os.environ.get(name)
    print(f"{name}: {'set' if value else 'not set'}")


def main() -> None:
    load_dotenv()
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
        env_status(name)

    if shutil.which("huggingface-cli"):
        run(["huggingface-cli", "whoami"])
    else:
        print("huggingface-cli is not installed. Install local launcher dependencies with: python -m pip install -r requirements-launch.txt")

    if shutil.which("modal"):
        run(["modal", "token", "current"])
    else:
        print("modal is not installed. Install local launcher dependencies with: python -m pip install -r requirements-launch.txt")


if __name__ == "__main__":
    main()
