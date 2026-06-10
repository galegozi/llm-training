# Chess LLM fine-tuning on Modal

This repo trains a chess move-selection LLM with QLoRA on Modal and uploads the result to Hugging Face.

## Current plan

- **Base model:** `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
  - This is the text-only Nemotron Nano 30B-A3B checkpoint, not the Omni variant.
  - It is under the requested 32B-parameter ceiling: 30B total parameters with 3.5B active parameters.
- **Dataset:** `espressocheese/LichessGames`
  - The dataset viewer shows columns including `whiteElo`, `blackElo`, `sysPlayer`, and `conversations`.
  - Rows are already chat-style chess move-selection examples.
- **Upload target:** `espressocheese/chess-llm`
- **Visibility:** public by default.

## What I still need from you

Nothing besides authentication. The pipeline is set up so you do not need to provide extra training choices before the first real run.

You need local access to:

1. **Hugging Face**: a token that can create/update `espressocheese/chess-llm` and upload model files.
2. **Modal**: a configured Modal account/token with credits available.

Do not paste tokens into the repo. The helper script reads your local Hugging Face login, creates/updates the public Hugging Face repo, creates the Modal secret, and launches the job.

## Recommended first full run

You said you have **$218.05** available and can add **$29**, for a likely total of **$247.05**. The default launch is intentionally a serious run, but still bounded:

```bash
python scripts/prepare_and_launch.py
```

Defaults used by that command:

- `--gpu H100`
- `--max-steps 4000`
- `--max-samples 500000`
- `--max-seq-length 1024`
- CPU-only dataset preprocessing/tokenization runs first and saves to the Modal volume before the H100 training container starts
- public Hugging Face upload to `espressocheese/chess-llm`
- LoRA adapter upload, not a merged 30B checkpoint

The adapter-first choice is deliberate: it is cheaper, faster, and safer for the first run. A merged full checkpoint can be launched later with `--merge-and-upload`, but that uses more memory, storage, and upload time.

## One-time setup

Install the local launcher dependencies only:

```bash
python -m pip install -r requirements-launch.txt
```

`requirements.txt` is still used inside the Modal training image and includes GPU/training packages such as PyTorch, Transformers, TRL, PEFT, and bitsandbytes. For your local machine or this agent runtime, `requirements-launch.txt` is enough to authenticate and start the Modal job.

If Hugging Face is not already configured locally:

```bash
huggingface-cli login
```

If Modal is not already configured locally:

```bash
modal setup
```

Check both without printing secrets:

```bash
python scripts/check_auth.py
```

If your environment supports file-based secret injection instead of exported variables, you can also create a local `.env` file with `HF_TOKEN`, `MODAL_TOKEN_ID`, and `MODAL_TOKEN_SECRET`. The helper scripts load `.env` automatically, and `.gitignore` prevents it from being committed.

## If package installation is blocked

If `pip install` fails with a proxy error such as `Tunnel connection failed: 403 Forbidden`, the runtime cannot download Python packages from PyPI. Fix one of these before launching from that runtime:

- allow outbound HTTPS/PyPI access for the environment,
- preinstall `huggingface_hub` and `modal`,
- inject a working package index/proxy configuration, or
- run `python scripts/prepare_and_launch.py` locally from a machine that already has working network access and Modal/Hugging Face credentials.

The Modal GPU container will install the full training dependencies remotely when the Modal job starts, so the local environment only needs the launcher dependencies and network access to Hugging Face/Modal.

## Hands-off launch

Run this after local Hugging Face and Modal auth are configured:

```bash
python scripts/prepare_and_launch.py
```

That command will:

1. Confirm Modal auth.
2. Read your local Hugging Face token without printing it.
3. Create or update `espressocheese/chess-llm` as a **public** model repo.
4. Store the Hugging Face token in a Modal secret named `huggingface`.
5. Launch a CPU-only Modal preprocessing job.
6. Stream, render, tokenize, truncate, and save the dataset to the shared Modal volume.
7. Start the GPU training job only after preprocessing completes.
8. Load the preprocessed dataset from the Modal volume.
9. Train the QLoRA adapter.
10. Upload the adapter, tokenizer, and model card to Hugging Face.

## Dry-run setup without launching training

If you only want to verify repo/secret setup first:

```bash
python scripts/prepare_and_launch.py --skip-launch
```

## Dataset preprocessing before GPU training

The default Modal entrypoint deliberately splits the run into two stages:

1. `preprocess_dataset` has **no GPU attached**. It streams `espressocheese/LichessGames`, applies the Nemotron tokenizer chat template, tokenizes/truncates examples to `--max-seq-length`, and saves the finite dataset to `/vol/preprocessed/chess-llm/train`.
2. `train` requests the configured GPU only after the preprocessing function returns successfully, then loads that saved dataset with `datasets.load_from_disk`.

If a preprocessing run already completed and you only want to restart training from the cached dataset, pass `--skip-preprocessing` to `scripts/prepare_and_launch.py` or the lower-level Modal command.

## Manual Modal launch

If the Modal secret already exists and you want to launch directly:

```bash
modal run modal_train.py --max-steps 4000 --max-samples 500000
```

Use another Modal GPU type by setting `MODAL_GPU` before launching:

```bash
MODAL_GPU=A100 modal run modal_train.py --max-steps 1000
```

## Important knobs

- `--max-steps`: hard cost-control cap. Default is `4000`.
- `--max-samples`: number of streamed examples to consume. Default is `500000`.
- `--max-seq-length`: default `1024`; used during CPU preprocessing/tokenization and training. Increase only if legal-move prompts are being truncated too much.
- `--gpu`: Modal GPU type used by `scripts/prepare_and_launch.py`; default is `H100`.
- `--private`: available on the lower-level Modal/training scripts, but not used by default because the target repo should be public.
- `--merge-and-upload`: optional full merge. Skip this for the first run unless you explicitly want a much larger upload.
- `--skip-preprocessing`: reuse `/vol/preprocessed/chess-llm/train` instead of rebuilding it. Use only after a successful preprocessing run.

## Files

- `modal_train.py`: Modal app and remote entrypoint.
- `training/preprocess_dataset.py`: CPU-only dataset rendering/tokenization script.
- `training/train_qlora.py`: QLoRA supervised fine-tuning script.
- `scripts/prepare_and_launch.py`: one-command repo setup, Modal secret setup, and training launch.
- `scripts/check_auth.py`: local auth sanity check.
- `requirements.txt`: full Python dependencies for the Modal training image.
- `requirements-launch.txt`: minimal local dependencies for auth setup and launching Modal.
