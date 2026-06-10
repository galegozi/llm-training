# LLM fine-tuning launcher

This repository contains a small OpenAI fine-tuning launcher that validates a local
JSONL dataset, uploads it with purpose `fine-tune`, and creates a fine-tuning job.
It follows the current OpenAI API flow: build a dataset, upload training data, and
create a fine-tuning job with a base model, uploaded training file, and method.

## Required inputs

Before starting a job, provide:

1. `OPENAI_API_KEY` with access to OpenAI fine-tuning.
2. A JSONL training file. By default, the script looks for `data/train.jsonl`.
3. Optionally, a JSONL validation file.

The default validator expects supervised chat-format examples such as:

```jsonl
{"messages":[{"role":"system","content":"Answer concisely."},{"role":"user","content":"Hello"},{"role":"assistant","content":"Hi!"}]}
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

If you only need to launch jobs and already have the package on `PYTHONPATH`, install the runtime dependency with `python -m pip install -r requirements.txt`.

## Dry run

Validate your local configuration and data without calling the API:

```bash
OPENAI_API_KEY=sk-... \
python -m fine_tuning.start_openai_fine_tune \
  --training-file data/train.jsonl \
  --model gpt-4o-mini-2024-07-18 \
  --suffix my-run \
  --dry-run
```

## Start a fine-tuning job

```bash
OPENAI_API_KEY=sk-... \
python -m fine_tuning.start_openai_fine_tune \
  --training-file data/train.jsonl \
  --validation-file data/validation.jsonl \
  --model gpt-4o-mini-2024-07-18 \
  --suffix my-run \
  --metadata project=llm-training
```

The command prints the fine-tuning job ID, initial status, uploaded file IDs, and
fine-tuned model field if one is already available.

## Environment variables

CLI flags take precedence, but the launcher also supports:

- `OPENAI_FINE_TUNE_TRAINING_FILE`
- `OPENAI_FINE_TUNE_VALIDATION_FILE`
- `OPENAI_FINE_TUNE_MODEL`
- `OPENAI_FINE_TUNE_METHOD`
- `OPENAI_FINE_TUNE_SUFFIX`

## Current environment status

At the time this scaffold was added, this workspace did not contain a training
JSONL file and did not have `OPENAI_API_KEY` set, so an actual fine-tuning job
could not be started from inside the container.
