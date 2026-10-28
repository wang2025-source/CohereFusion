"""Fuse registered infrared and visible images with CohereFusion."""

import argparse
from contextlib import nullcontext
from pathlib import Path

import clip
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from tqdm import tqdm

from data.io import paired_images, descriptions, image_description
from data.tasks import TASKS, TASK_KEYS, DEFAULT_PROMPTS, GENERIC_PROMPT
from model import CohereFusion
from scripts.checkpoint import load_checkpoint


def input_groups(args):
    if args.visible_dir or args.infrared_dir:
        if not (args.visible_dir and args.infrared_dir) or args.dataset_root:
            raise ValueError("Use either --dataset-root or both --visible-dir and --infrared-dir")
        return [("", Path(args.visible_dir), Path(args.infrared_dir), GENERIC_PROMPT)]
    if not args.dataset_root:
        raise ValueError("Provide --dataset-root or explicit visible/infrared directories")
    root = Path(args.dataset_root)
    if all((root / task).is_dir() for task in TASKS):
        return [(task, root / task / "test" / "Visible", root / task / "test" / "Infrared",
                 DEFAULT_PROMPTS[TASK_KEYS[task]]) for task in TASKS]
    for parent in (root, root / "test"):
        for visible, infrared in (("Visible", "Infrared"), ("vi", "ir"), ("visible", "infrared")):
            if (parent / visible).is_dir() and (parent / infrared).is_dir():
                return [("", parent / visible, parent / infrared, GENERIC_PROMPT)]
    raise ValueError(f"No registered source directories found below {root}")


def main(args):
    groups = [(task, paired_images(visible, infrared), default)
              for task, visible, infrared, default in input_groups(args)]
    prompts = descriptions(args.text_file) if args.text_file else {}
    if args.prompt is not None and not args.prompt.strip():
        raise ValueError("Custom prompt must contain non-whitespace text")
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("args"), dict):
        raise ValueError("Checkpoint must contain architecture, model and training arguments")
    if checkpoint.get("architecture") != CohereFusion.architecture:
        raise ValueError("Expected a CohereFusion checkpoint")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu")
    config = checkpoint["args"]
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    model = CohereFusion(clip_model, dim=config.get("dim", 32),
                         global_top_k=config.get("global_top_k", 4)).to(device)
    del clip_model
    load_checkpoint(model, checkpoint)
    model.eval()
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    for task, pairs, default in groups:
        destination = Path(args.output) / task
        destination.mkdir(parents=True, exist_ok=True)
        for visible_path, infrared_path in tqdm(pairs, desc=task or "Fusion"):
            with Image.open(visible_path) as image:
                visible = TF.to_tensor(image.convert("RGB"))[None].to(device)
            with Image.open(infrared_path) as image:
                infrared = TF.to_tensor(image.convert("RGB"))[None].to(device)
            if visible.shape != infrared.shape:
                raise ValueError(f"Mismatched registered pair dimensions: {visible_path}")
            height, width = visible.shape[-2:]
            padding = (0, (-width) % 8, 0, (-height) % 8)
            visible = F.pad(visible, padding, mode="replicate")
            infrared = F.pad(infrared, padding, mode="replicate")
            prompt = image_description(prompts, task, visible_path, None) if args.text_file else (args.prompt or default)
            tokens = clip.tokenize([prompt], truncate=True).to(device)
            precision = (torch.autocast(device_type="cuda", dtype=amp_dtype)
                         if device.type == "cuda" and args.precision == "amp" else nullcontext())
            with torch.inference_mode(), precision:
                fused = model(visible, infrared, tokens).float()[..., :height, :width].clamp(0, 1)
            TF.to_pil_image(fused[0].cpu()).save(destination / f"{visible_path.stem}.png")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root")
    parser.add_argument("--visible-dir")
    parser.add_argument("--infrared-dir")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", default="./outputs/coherefusion")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "amp"), default="fp32")
    text = parser.add_mutually_exclusive_group()
    text.add_argument("--prompt")
    text.add_argument("--text-file")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
