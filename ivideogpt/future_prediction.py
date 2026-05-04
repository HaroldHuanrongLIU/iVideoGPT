from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image

from benchmark.surgwmbench import BaselineSpec, run_cli


def predict_native_ivideogpt_frames(
    dataset_root: Path,
    window: object,
    args: argparse.Namespace,
) -> List[np.ndarray]:
    checkpoint = Path(str(args.checkpoint))
    if checkpoint.suffix == ".json":
        raise RuntimeError(
            "iVideoGPT native prediction requires a real pretrained model directory or "
            "Hugging Face model id, not the adapter metadata checkpoint: "
            f"{checkpoint}"
        )
    if checkpoint.exists() and not checkpoint.is_dir():
        raise RuntimeError(
            "iVideoGPT native prediction expects a pretrained model directory with "
            "'tokenizer' and 'transformer' subfolders, or a Hugging Face model id. "
            f"Got file checkpoint: {checkpoint}"
        )
    if checkpoint.suffix in {".pt", ".pth", ".ckpt", ".safetensors"}:
        raise RuntimeError(
            "iVideoGPT native prediction expects a pretrained model directory or "
            f"Hugging Face model id, not a single checkpoint file: {checkpoint}"
        )

    try:
        import torch
        from transformers import AutoModelForCausalLM

        from ivideogpt.vq_model import CompressiveVQModel
    except Exception as exc:  # pragma: no cover - depends on the native iVideoGPT env.
        raise RuntimeError(
            "iVideoGPT native prediction requires the upstream iVideoGPT runtime "
            "dependencies, including torch, transformers, diffusers, and safetensors."
        ) from exc

    device = str(args.device)
    model_ref = str(args.checkpoint)
    tokenizer = getattr(args, "_native_ivideogpt_tokenizer", None)
    model = getattr(args, "_native_ivideogpt_model", None)
    if tokenizer is None or model is None:
        tokenizer = CompressiveVQModel.from_pretrained(
            model_ref,
            subfolder="tokenizer",
            low_cpu_mem_usage=False,
        ).to(device)
        if int(getattr(tokenizer, "context_length", -1)) != int(args.context_frames):
            tokenizer.set_context_length(int(args.context_frames))
        model = AutoModelForCausalLM.from_pretrained(
            model_ref,
            subfolder="transformer",
            low_cpu_mem_usage=False,
        ).to(device)
        model.eval()
        expected_vocab = tokenizer.num_vq_embeddings + tokenizer.num_dyn_embeddings + 2
        if int(getattr(model.config, "vocab_size", -1)) != int(expected_vocab):
            raise RuntimeError(
                "iVideoGPT transformer/tokenizer vocabulary mismatch: "
                f"{getattr(model.config, 'vocab_size', None)} vs {expected_vocab}"
            )
        setattr(args, "_native_ivideogpt_tokenizer", tokenizer)
        setattr(args, "_native_ivideogpt_model", model)

    native_resolution = int(getattr(tokenizer, "config", {}).get("resolution", args.image_size))
    context_frames: List[np.ndarray] = []
    for frame_path in window.context_frame_paths[: int(args.context_frames)]:
        with Image.open(dataset_root / frame_path) as image:
            image = image.convert("RGB").resize((native_resolution, native_resolution))
            context_frames.append(np.asarray(image, dtype=np.uint8))
    if len(context_frames) != int(args.context_frames):
        raise RuntimeError(
            f"Expected {args.context_frames} context frames, got {len(context_frames)}"
        )

    dummy_future = [context_frames[-1].copy() for _ in range(int(args.prediction_horizon))]
    video = np.stack([*context_frames, *dummy_future], axis=0)
    pixel_values = torch.from_numpy(video).permute(0, 3, 1, 2).unsqueeze(0).float()
    pixel_values = (pixel_values / 255.0).to(device)

    with torch.inference_mode():
        tokens, _ = tokenizer.tokenize(pixel_values, int(args.context_frames))
        context_tokens_per_frame = 16 * 16 + 1
        gen_input = tokens[:, : int(args.context_frames) * context_tokens_per_frame]
        max_new_tokens = (1 + 4 * 4) * int(args.prediction_horizon) - 1
        generated_tokens = model.generate(
            gen_input,
            do_sample=True,
            temperature=1.0,
            top_k=100,
            max_new_tokens=max_new_tokens,
            pad_token_id=50256,
        )
        recon_output = tokenizer.detokenize(generated_tokens, int(args.context_frames)).clamp(0.0, 1.0)

    available_future = int(recon_output.shape[1]) - int(args.context_frames)
    if available_future < int(args.prediction_horizon):
        raise RuntimeError(
            "iVideoGPT native checkpoint generated too few future frames: "
            f"available {available_future}, requested {args.prediction_horizon}"
        )
    frames: List[np.ndarray] = []
    for offset in range(int(args.prediction_horizon)):
        frame_index = int(args.context_frames) + offset
        frame = recon_output[0, frame_index].detach().cpu().permute(1, 2, 0).numpy()
        frame_uint8 = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        if frame_uint8.shape[:2] != (int(args.image_size), int(args.image_size)):
            frame_uint8 = np.asarray(
                Image.fromarray(frame_uint8).resize((int(args.image_size), int(args.image_size)))
            )
        frames.append(frame_uint8)
    return frames


SPEC = BaselineSpec(
    baseline="ivideogpt",
    model="iVideoGPT",
    native_entrypoint="inference/predict.py",
    native_train_entrypoint="train_gpt.py",
    native_frame_predictor=(
        "native SurgWMBench sparse future-frame prediction is wired through the "
        "iVideoGPT tokenizer and transformer when --checkpoint points to a pretrained "
        "model directory or Hugging Face model id."
    ),
    native_frame_predictor_fn=predict_native_ivideogpt_frames,
    notes=(
        "The adapter uses official SurgWMBench manifests directly. CPU smoke runs can "
        "use copy-last frames; native frame prediction requires iVideoGPT dependencies "
        "and a compatible action-free pretrained checkpoint. Trajectory outputs use the "
        "shared deterministic trajectory head unless a model-specific head is added."
    ),
)


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_cli(SPEC, argv)


if __name__ == "__main__":
    main()
