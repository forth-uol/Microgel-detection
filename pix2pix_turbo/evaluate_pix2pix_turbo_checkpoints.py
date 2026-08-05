#!/usr/bin/env python3
"""Evaluate every pix2pix-turbo checkpoint on a paired test set.

This script is intended to run from a GPU job.  It loads checkpoints one at a
time, writes generated images and A|prediction|B comparisons, and creates a
CSV summary containing paired reconstruction metrics and (optionally) FID.
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import re
import sys
from pathlib import Path

import lpips
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_pil_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turbo-root", type=Path, required=True)
    parser.add_argument("--dataset-folder", type=Path, required=True)
    parser.add_argument("--checkpoints-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-prep", default="no_resize")
    parser.add_argument("--max-samples", type=int, default=-1,
                        help="-1 evaluates the complete test set")
    parser.add_argument("--comparison-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-fid", action="store_true")
    return parser.parse_args()


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"model_(\d+)\.pkl", path.name)
    if match is None:
        raise ValueError(f"Unexpected checkpoint filename: {path.name}")
    return int(match.group(1))


def gaussian_window(channels: int, device: torch.device) -> torch.Tensor:
    coords = torch.arange(11, dtype=torch.float32, device=device) - 5
    kernel = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    kernel = kernel / kernel.sum()
    window = (kernel[:, None] @ kernel[None, :]).view(1, 1, 11, 11)
    return window.expand(channels, 1, 11, 11).contiguous()


def ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Standard Gaussian-window SSIM for tensors in [0, 1]."""
    channels = pred.shape[1]
    window = gaussian_window(channels, pred.device)
    mu_x = F.conv2d(pred, window, padding=5, groups=channels)
    mu_y = F.conv2d(target, window, padding=5, groups=channels)
    mu_x2, mu_y2 = mu_x.square(), mu_y.square()
    mu_xy = mu_x * mu_y
    sigma_x2 = F.conv2d(pred.square(), window, padding=5, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(target.square(), window, padding=5, groups=channels) - mu_y2
    sigma_xy = F.conv2d(pred * target, window, padding=5, groups=channels) - mu_xy
    score = ((2 * mu_xy + 0.01 ** 2) * (2 * sigma_xy + 0.03 ** 2)) / (
        (mu_x2 + mu_y2 + 0.01 ** 2) * (sigma_x2 + sigma_y2 + 0.03 ** 2)
    )
    return score.flatten(1).mean(1)


def save_comparison(source: torch.Tensor, pred: torch.Tensor,
                    target: torch.Tensor, destination: Path) -> None:
    images = [to_pil_image(x.detach().cpu().clamp(0, 1))
              for x in (source, pred, target)]
    canvas = Image.new("RGB", (sum(im.width for im in images), images[0].height))
    left = 0
    for image in images:
        canvas.paste(image.convert("RGB"), (left, 0))
        left += image.width
    canvas.save(destination)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")

    required = [args.dataset_folder / "test_A",
                args.dataset_folder / "test_B",
                args.dataset_folder / "test_prompts.json"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing test data: " + ", ".join(missing))

    checkpoints = sorted(args.checkpoints_dir.glob("model_*.pkl"), key=checkpoint_step)
    if not checkpoints:
        raise FileNotFoundError(f"No model_*.pkl files in {args.checkpoints_dir}")

    sys.path.insert(0, str(args.turbo_root / "src"))
    from my_utils.training_utils import PairedDataset  # noqa: PLC0415
    from pix2pix_turbo import Pix2Pix_Turbo  # noqa: PLC0415

    device = torch.device("cuda")
    perceptual = lpips.LPIPS(net="vgg").to(device).eval()
    perceptual.requires_grad_(False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "metrics_summary.csv"
    fields = ["step", "samples", "l1", "mse", "psnr_db", "ssim", "lpips", "clean_fid"]

    completed = set()
    if summary_path.exists():
        with summary_path.open(newline="", encoding="utf-8") as handle:
            completed = {int(row["step"]) for row in csv.DictReader(handle)}

    for checkpoint in checkpoints:
        step = checkpoint_step(checkpoint)
        if step in completed:
            print(f"Skipping completed checkpoint {step}", flush=True)
            continue

        print(f"Evaluating {checkpoint}", flush=True)
        model = Pix2Pix_Turbo(pretrained_path=str(checkpoint))
        model.set_eval()
        dataset = PairedDataset(
            dataset_folder=str(args.dataset_folder), split="test",
            image_prep=args.image_prep, tokenizer=model.tokenizer,
        )
        sample_count = len(dataset) if args.max_samples < 0 else min(len(dataset), args.max_samples)
        step_dir = args.output_dir / f"step_{step}"
        generated_dir = step_dir / "generated"
        comparisons_dir = step_dir / "comparisons"
        generated_dir.mkdir(parents=True, exist_ok=True)
        comparisons_dir.mkdir(parents=True, exist_ok=True)

        totals = {name: 0.0 for name in ("l1", "mse", "psnr_db", "ssim", "lpips")}
        with torch.inference_mode():
            for index in range(sample_count):
                sample = dataset[index]
                # Reset the seed per image so every checkpoint receives identical VAE noise.
                torch.manual_seed(args.seed + index)
                torch.cuda.manual_seed_all(args.seed + index)
                source = sample["conditioning_pixel_values"].unsqueeze(0).to(device)
                target_m11 = sample["output_pixel_values"].unsqueeze(0).to(device)
                prompt_tokens = sample["input_ids"].to(device)
                pred_m11 = model(source, prompt_tokens=prompt_tokens, deterministic=True)
                pred = (pred_m11.float() * 0.5 + 0.5).clamp(0, 1)
                target = (target_m11.float() * 0.5 + 0.5).clamp(0, 1)

                mse_value = F.mse_loss(pred, target).item()
                totals["l1"] += F.l1_loss(pred, target).item()
                totals["mse"] += mse_value
                totals["psnr_db"] += -10.0 * math.log10(max(mse_value, 1e-12))
                totals["ssim"] += ssim(pred, target).item()
                totals["lpips"] += perceptual(pred_m11.float(), target_m11.float()).mean().item()

                name = Path(dataset.img_names[index]).stem + ".png"
                to_pil_image(pred[0].cpu()).save(generated_dir / name)
                if index < args.comparison_count:
                    save_comparison(source[0], pred[0], target[0], comparisons_dir / name)
                if (index + 1) % 25 == 0 or index + 1 == sample_count:
                    print(f"  {index + 1}/{sample_count}", flush=True)

        fid_value = ""
        if not args.skip_fid:
            from cleanfid import fid  # noqa: PLC0415
            fid_value = fid.compute_fid(
                str(generated_dir), str(args.dataset_folder / "test_B"),
                mode="clean", device="cuda", num_workers=0, batch_size=8,
            )

        row = {"step": step, "samples": sample_count, "clean_fid": fid_value}
        row.update({name: value / sample_count for name, value in totals.items()})
        write_header = not summary_path.exists()
        with summary_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        print(f"Completed step {step}: {row}", flush=True)

        del dataset, model
        gc.collect()
        torch.cuda.empty_cache()

    print(f"All results: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
