# Repository Guidelines

## Project Structure & Module Organization

`ivideogpt/` contains the core model code: data loaders in `ivideogpt/data/`, tokenizer/VQ modules in `ivideogpt/vq_model/`, transformer heads in `ivideogpt/transformer/`, and metrics in `ivideogpt/utils/`. Top-level training entrypoints are `train_tokenizer.py` and `train_gpt.py`. `inference/` holds prediction utilities and sample `.npz` inputs. `datasets/` contains dataset conversion and preprocessing scripts, with dataset paths configured in `DATASET.yaml`. `scripts/pretrain/`, `scripts/finetune/`, and `scripts/evaluation/` provide runnable experiment shells. `mbrl/` contains visual model-based RL code and Hydra configs, while `vp/` contains Visual Planning integration files. Keep images and diagrams in `assets/`.

## Build, Test, and Development Commands

Install the base environment with:

```bash
conda create -n ivideogpt python==3.9
conda activate ivideogpt
pip install -r requirements.txt
```

Run a prediction smoke test with:

```bash
python inference/predict.py --pretrained_model_name_or_path thuml/ivideogpt-oxe-64-act-free --input_path inference/samples/fractal_sample.npz --dataset_name fractal20220817_data
```

Use provided experiment scripts rather than duplicating long commands, for example `bash scripts/pretrain/oxe-64-act-free.sh`, `bash scripts/finetune/bair-64-act-cond.sh`, or `bash scripts/evaluation/bair-64-act-cond.sh`. For syntax checks, run `python -m compileall ivideogpt inference datasets mbrl`.

## Coding Style & Naming Conventions

Use Python with 4-space indentation. Follow the existing module style: lowercase snake_case for files, functions, variables, and CLI flags; PascalCase for model and dataset classes. Keep config names descriptive and resolution/task-oriented, such as `bair-64-act-cond.sh` or `configs/llama/config_medium.json`. Prefer explicit paths and arguments in scripts; avoid hidden global state beyond the existing config files.

## Testing Guidelines

This repository currently has no dedicated test suite. Validate changes with the smallest relevant command: `compileall` for import/syntax changes, `inference/predict.py` for inference changes, and the matching shell script under `scripts/` for training or evaluation changes. For data changes, test preprocessing on a tiny subset before editing `DATASET.yaml`.

## Commit & Pull Request Guidelines

Git history uses short, direct messages such as `Update README.md`, `fix bugs in video planning tasks`, and `add vp files`. Keep commits focused and use imperative, concise subjects. Pull requests should describe the changed component, list commands run, mention required datasets/checkpoints, and include sample outputs or screenshots when prediction behavior or visual results change.

## Security & Configuration Tips

Do not commit local dataset roots, downloaded checkpoints, generated logs, or experiment outputs. Keep machine-specific paths in `DATASET.yaml`, `mbrl/cfgs/*.yaml`, or local scripts, and document any required external model path such as `pretrained_models/i3d/i3d_torchscript.pt`.
