import argparse
import json
import math
import sys
from pathlib import Path

import imageio
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ivideogpt.data import DEFAULT_SURGWMBENCH_ROOT, SurgWMBenchAnchorDataset
from ivideogpt.vq_model import CompressiveVQModel


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate iVideoGPT on SurgWMBench 20-anchor prediction.")
    parser.add_argument("--surgwmbench_root", default=DEFAULT_SURGWMBENCH_ROOT)
    parser.add_argument("--manifest", default="manifests/test.jsonl")
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--transformer_path", required=True)
    parser.add_argument("--output_dir", default="benchmark/outputs/ivideogpt_surgwmbench_anchor")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--context_length", type=int, default=5)
    parser.add_argument("--segment_length", type=int, default=20)
    parser.add_argument("--context_token_grid", type=int, default=16)
    parser.add_argument("--future_token_grid", type=int, default=4)
    parser.add_argument("--max_clips", type=int, default=None)
    parser.add_argument("--num_artifacts", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument(
        "--metric_resolution",
        choices=["original", "model"],
        default="original",
        help="'original': resize predictions to original frame size and score there. "
             "'model': downsample GT to model resolution and score there (no upsample).",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_original_frame(dataset_root: Path, frame_path: str) -> torch.Tensor:
    with Image.open(dataset_root / frame_path) as image:
        image = image.convert("RGB")
        return F.to_tensor(image)


def resize_prediction_to_original(prediction: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    height, width = original.shape[-2:]
    return F.resize(
        prediction,
        [height, width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).clamp(0.0, 1.0)


def downsample_gt_to_model_resolution(gt: torch.Tensor, resolution: int) -> torch.Tensor:
    return F.resize(
        gt,
        [resolution, resolution],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).clamp(0.0, 1.0)


def tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    return (tensor.permute(1, 2, 0).detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)


def mean_records(records):
    if not records:
        return {}
    keys = [key for key in records[0].keys() if isinstance(records[0][key], (int, float))]
    return {key: float(np.mean([record[key] for record in records])) for key in keys}


def summarize(records):
    overall = mean_records(records)
    by_difficulty = {}
    for difficulty in sorted({record["difficulty"] for record in records}):
        by_difficulty[difficulty] = mean_records([record for record in records if record["difficulty"] == difficulty])
    return {"overall": overall, "by_difficulty": by_difficulty, "num_clips": len(records)}


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    dataset_root = Path(args.surgwmbench_root)
    output_dir = Path(args.output_dir)
    artifact_dir = output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    dataset = SurgWMBenchAnchorDataset(
        dataset_root=dataset_root,
        manifest=args.manifest,
        image_size=args.resolution,
        max_samples=args.max_clips,
        return_metadata=True,
    )

    tokenizer = CompressiveVQModel.from_pretrained(
        args.tokenizer_path,
        subfolder=None,
        revision=None,
        variant=None,
        use_safetensor=True,
        low_cpu_mem_usage=False,
        device_map=None,
    ).eval().to(device)
    if tokenizer.context_length != args.context_length:
        tokenizer.set_context_length(args.context_length)

    model = AutoModelForCausalLM.from_pretrained(args.transformer_path).eval().to(device)
    expected_vocab_size = tokenizer.num_vq_embeddings + tokenizer.num_dyn_embeddings + 2
    if model.config.vocab_size != expected_vocab_size:
        raise ValueError(
            f"Transformer vocab size {model.config.vocab_size} does not match tokenizer "
            f"vocab size {expected_vocab_size}."
        )

    try:
        import lpips
        import piqa
    except ImportError as exc:
        raise ImportError("Install piqa and lpips to run original-resolution image metrics.") from exc

    mse_loss = torch.nn.MSELoss()
    psnr_metric = piqa.PSNR(epsilon=1e-8, value_range=1.0, reduction="none").to(device)
    ssim_metric = piqa.SSIM(window_size=11, sigma=1.5, n_channels=3, reduction="none").to(device)
    lpips_metric = lpips.LPIPS(net="vgg").eval().to(device)

    horizons = {
        "horizon_5": 5,
        "horizon_10": 10,
        "horizon_15": 15,
    }
    records_by_horizon = {name: [] for name in horizons}
    sample_artifacts = []

    context_token_count = args.context_length * (1 + args.context_token_grid ** 2)
    max_new_tokens = (1 + args.future_token_grid ** 2) * (args.segment_length - args.context_length) - 1

    for clip_idx in tqdm(range(len(dataset)), desc="surgwmbench-anchor-eval"):
        sample = dataset[clip_idx]
        pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
        metadata = sample["metadata"]

        tokens, _ = tokenizer.tokenize(pixel_values, args.context_length)
        gen_input = tokens[:, :context_token_count]
        generated_tokens = model.generate(
            gen_input,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_k=args.top_k,
            max_new_tokens=max_new_tokens,
            pad_token_id=50256,
        )
        prediction = tokenizer.detokenize(generated_tokens, args.context_length).clamp(0.0, 1.0).cpu()

        per_future_frame = []
        for target_offset in range(args.segment_length - args.context_length):
            anchor_idx = args.context_length + target_offset
            gt = load_original_frame(dataset_root, metadata["anchor_frame_paths"][anchor_idx]).to(device)
            if args.metric_resolution == "original":
                pred = resize_prediction_to_original(prediction[0, anchor_idx].to(device), gt)
                gt_for_metric = gt
            else:
                pred = prediction[0, anchor_idx].to(device).clamp(0.0, 1.0)
                gt_for_metric = downsample_gt_to_model_resolution(gt, args.resolution)
            gt_batch = gt_for_metric.unsqueeze(0)
            pred_batch = pred.unsqueeze(0)
            mse = mse_loss(pred_batch, gt_batch).item()
            psnr = psnr_metric(pred_batch, gt_batch).mean().item()
            ssim = ssim_metric(pred_batch, gt_batch).mean().item()
            lpips_value = lpips_metric(pred_batch * 2.0 - 1.0, gt_batch * 2.0 - 1.0).mean().item()
            per_future_frame.append({
                "mse": mse,
                "psnr": psnr if math.isfinite(psnr) else float("inf"),
                "ssim": ssim,
                "lpips": lpips_value,
            })

        for horizon_name, horizon in horizons.items():
            horizon_metrics = mean_records(per_future_frame[:horizon])
            horizon_metrics.update({
                "difficulty": metadata["difficulty"],
                "patient_id": metadata["patient_id"],
                "trajectory_id": metadata["trajectory_id"],
            })
            records_by_horizon[horizon_name].append(horizon_metrics)

        if clip_idx < args.num_artifacts:
            for horizon_name, horizon in horizons.items():
                frames = []
                for target_offset in range(horizon):
                    anchor_idx = args.context_length + target_offset
                    gt = load_original_frame(dataset_root, metadata["anchor_frame_paths"][anchor_idx])
                    pred = resize_prediction_to_original(prediction[0, anchor_idx], gt)
                    frames.append(np.concatenate([tensor_to_uint8_image(gt), tensor_to_uint8_image(pred)], axis=1))
                artifact_path = artifact_dir / (
                    f"{metadata['patient_id']}-{metadata['trajectory_id']}-{horizon_name}.gif"
                )
                imageio.mimsave(artifact_path, frames, fps=4, loop=0)
                sample_artifacts.append(str(artifact_path))

    metrics = {
        "dataset_name": "SurgWMBench",
        "prediction_task": "20_anchor_video_prediction",
        "manifest": args.manifest,
        "tokenizer_path": args.tokenizer_path,
        "transformer_path": args.transformer_path,
        "model_resolution": [args.resolution, args.resolution],
        "original_resolution": [1920, 1080],
        "context_anchor_count": args.context_length,
        "target_horizons": {
            "horizon_5": "anchors 6-10",
            "horizon_10": "anchors 6-15",
            "horizon_15": "anchors 6-20",
        },
        "metric_resolution": args.metric_resolution,
        "resize_policy": (
            "full-frame bicubic resize to model resolution; predictions bicubic-resized back to original frame size for metrics"
            if args.metric_resolution == "original"
            else f"full-frame bicubic resize to model resolution; GT bicubic-downsampled to {args.resolution}x{args.resolution} for metrics (predictions kept at model resolution)"
        ),
        "num_clips": len(dataset),
        "metrics": {name: summarize(records) for name, records in records_by_horizon.items()},
        "sample_artifacts": sample_artifacts,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
