# Usage

This guide covers the SurgWMBench 20-anchor workflow for this fork: create the
Python environment, train on one or more GPUs, and evaluate predictions against
the original-resolution frames.

## Environment

Install dependencies with `uv`:

```bash
uv sync
source .venv/bin/activate
```

The environment uses Python 3.11 and CUDA 13.0 PyTorch wheels from
`pyproject.toml`. TensorFlow and legacy dataset-conversion packages are not
installed by default.

Check the installed PyTorch build:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## Dataset

Set the dataset root before training or evaluation:

```bash
export SURGWMBENCH_ROOT=/mnt/hdd1/neurips2026_dataset_track/SurgWMBench
```

The SurgWMBench path uses official sparse 20-anchor manifests. Training uses the
first 5 anchors as context and predicts anchors 6-20.

## Single-GPU Training

The finetuning script runs two stages: tokenizer finetuning, then transformer
finetuning. It also downloads the required pretrained tokenizer and transformer
checkpoints if they are missing locally.

```bash
CUDA_VISIBLE_DEVICES=0 NUM_PROCESSES=1 bash scripts/finetune/surgwmbench-anchor-256.sh
```

Default outputs are written under:

```text
log_vqgan/
log_trm/
```

Useful overrides:

```bash
TOKENIZER_OUTPUT_ROOT=log_vqgan_run1 \
TRANSFORMER_OUTPUT_ROOT=log_trm_run1 \
CUDA_VISIBLE_DEVICES=0 NUM_PROCESSES=1 \
bash scripts/finetune/surgwmbench-anchor-256.sh
```

## Multi-GPU Training

Set `NUM_PROCESSES` to the number of GPUs to use. The training code prepares the
train and validation dataloaders with Accelerate so each process receives a
different shard.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PROCESSES=4 bash scripts/finetune/surgwmbench-anchor-256.sh
```

You can also select GPUs through the script-level `GPU_IDS` option:

```bash
GPU_IDS=0,1,2,3 NUM_PROCESSES=4 bash scripts/finetune/surgwmbench-anchor-256.sh
```

Keep the per-device batch size small for 256x256 training. The script defaults
to batch size 1 per GPU with gradient accumulation 4.

## Evaluation

Point `TOKENIZER_DIR` and `TRANSFORMER_DIR` to the trained checkpoint
directories. Evaluation predicts anchors 6-10, 6-15, and 6-20 from the first 5
anchors, resizes predictions back to each frame's original resolution, and
computes image metrics against the original target frames.

```bash
export TOKENIZER_DIR=log_vqgan/<timestamp>-surgwmbench_anchor_tokenizer_256
export TRANSFORMER_DIR=log_trm/<timestamp>-surgwmbench_anchor_transformer_256
export OUTPUT_DIR=benchmark/outputs/ivideogpt_surgwmbench_anchor_test

bash scripts/evaluation/surgwmbench-anchor-256.sh
```

The main metrics file is:

```text
benchmark/outputs/ivideogpt_surgwmbench_anchor_test/metrics.json
```

Visual samples, when enabled, are written under:

```text
benchmark/outputs/ivideogpt_surgwmbench_anchor_test/artifacts/
```

Limit evaluation for a quick smoke run:

```bash
python tools/evaluate_surgwmbench_anchor_prediction.py \
  --surgwmbench_root "${SURGWMBENCH_ROOT}" \
  --manifest manifests/test.jsonl \
  --tokenizer_path "${TOKENIZER_DIR}" \
  --transformer_path "${TRANSFORMER_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --resolution 256 \
  --context_length 5 \
  --segment_length 20 \
  --max_clips 2 \
  --num_artifacts 1
```
