"""Train CohereFusion with joint candidate supervision."""

import argparse
import hashlib
import json
import math
import os
import random
from datetime import datetime
from pathlib import Path

import clip
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataset import CohereEMSDataset
from data.tasks import DEFAULT_PROMPTS
from model.cohere_fusion import CohereFusion, ARCHITECTURE
from scripts.checkpoint import fusion_state, load_checkpoint
from scripts.losses import CohereFusionLoss


COUNTERFACTUAL = {"low_light": "ir_noise", "over_exposure": "ir_low_contrast",
                  "ir_low_contrast": "over_exposure", "ir_noise": "low_light"}


def schedule(step, total, warmup):
    if step < warmup:
        return max(1e-3, (step + 1) / max(1, warmup))
    progress = min(1., (step - warmup) / max(1, total - warmup))
    return .05 + .95 * .5 * (1 + math.cos(math.pi * progress))


def save_checkpoint(path, model, optimizer, scheduler, epoch, best, args, scaler=None):
    temporary = path.with_suffix(".tmp")
    state = np.random.get_state()
    rng = {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), *state[2:]],
           "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    split_path = getattr(args, "split_manifest", None)
    split_hash = hashlib.sha256(Path(split_path).read_bytes()).hexdigest() if split_path else None
    torch.save({"architecture": ARCHITECTURE, "model": fusion_state(model),
                "optimizer": optimizer.state_dict(), "lr_scheduler": scheduler.state_dict(),
                "epoch": epoch, "best_val_loss": best, "args": vars(args), "rng": rng,
                "split_sha256": split_hash,
                "scaler": scaler.state_dict() if scaler is not None else None}, temporary)
    os.replace(temporary, path)


def epoch_pass(model, loader, objective, device, epoch, optimizer=None, scheduler=None,
               *, scaler=None, grad_clip=1.):
    training = optimizer is not None
    model.train(training)
    if training:
        loader.dataset.set_epoch(epoch)
    totals, count = {}, 0
    amp = device.type == "cuda"
    dtype = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    if training and scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=amp and dtype == torch.float16)
    with torch.set_grad_enabled(training):
        for visible, infrared, visible_gt, infrared_gt, tasks, prompts, _ in tqdm(
                loader, desc=f"{'train' if training else 'val'} {epoch:03d}"):
            visible, infrared, visible_gt, infrared_gt = [
                tensor.to(device, non_blocking=True) for tensor in (visible, infrared, visible_gt, infrared_gt)]
            tokens = clip.tokenize(list(prompts), truncate=True).to(device)
            replacement = clip.tokenize([DEFAULT_PROMPTS[COUNTERFACTUAL[t]] for t in tasks], truncate=True).to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
                fused, aux = model(visible, infrared, tokens, counter_text=replacement, return_aux=True)
            # Soft histograms and normalization losses run in float32.
            loss, parts = objective(visible, infrared, visible_gt, infrared_gt, fused.float(),
                                    float_tensors(aux), tasks)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None and scaler.get_scale() >= previous_scale:
                    scheduler.step()
            batch = visible.shape[0]
            count += batch
            for name, value in {"loss": loss.detach(), **parts}.items():
                totals[name] = totals.get(name, 0.) + float(value) * batch
    if not count:
        raise ValueError("Data loader produced no batches")
    return {name: value / count for name, value in totals.items()}


def float_tensors(value):
    if isinstance(value, torch.Tensor):
        return value.float() if value.is_floating_point() else value
    if isinstance(value, dict):
        return {key: float_tensors(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(float_tensors(item) for item in value)
    return value


def main(args):
    if args.crop_size < 8 or args.crop_size % 8:
        raise ValueError("Crop size must be a multiple of 8")
    if min(args.epochs, args.batch_size, args.val_every) < 1 or args.workers < 0:
        raise ValueError("Epochs, batch size and validation interval must be positive; workers cannot be negative")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu")
    train_set = CohereEMSDataset(args.dataset_root, "train", args.split_manifest, args.crop_size, args.seed)
    val_set = CohereEMSDataset(args.dataset_root, "val", args.split_manifest, args.crop_size, args.seed)
    if len(train_set) < args.batch_size:
        raise ValueError("Training set has fewer samples than one full batch")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=device.type == "cuda", collate_fn=train_set.collate_fn, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.workers,
                            pin_memory=device.type == "cuda", collate_fn=val_set.collate_fn)
    print(f"Explicit scene-disjoint splits: train={train_set.counts}; val={val_set.counts}")
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    model = CohereFusion(clip_model, dim=args.dim, global_top_k=args.global_top_k).to(device)
    del clip_model
    branch_params, routing_params = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (routing_params if name.startswith(("rsfr.", "scr.")) else branch_params).append(parameter)
    optimizer = torch.optim.AdamW([{"params": branch_params, "lr": args.branch_lr},
                                   {"params": routing_params, "lr": args.router_lr}],
                                  betas=(.9, .99), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: schedule(
        step, args.epochs * len(train_loader), args.warmup_epochs * len(train_loader)))
    loss_keys = ("bins", "entropy_margin", "mi_floor", "scd_floor", "scd_weight", "text_weight",
                 "dynamic_weight", "dynamic_margin", "counterfactual_weight", "counterfactual_margin", "residual_weight",
                 "global_weight", "semantic_weight", "fixed_weight", "fixed_margin",
                 "diversity_weight", "diversity_margin", "fixed_targets", "fixed_importance",
                 "dynamic_rho", "fixed_rho")
    objective = CohereFusionLoss(**{key: getattr(args, key) for key in loss_keys}).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not torch.cuda.is_bf16_supported())
    start, best = 0, float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        load_checkpoint(model, checkpoint)
        split_hash = hashlib.sha256(Path(args.split_manifest).read_bytes()).hexdigest()
        if checkpoint.get("split_sha256") not in (None, split_hash):
            raise ValueError("Resume checkpoint uses a different scene partition")
        optimizer.load_state_dict(checkpoint["optimizer"]); scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start, best = checkpoint["epoch"] + 1, checkpoint["best_val_loss"]
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        rng = checkpoint.get("rng")
        if rng:
            random.setstate(rng["python"])
            state = rng["numpy"]
            np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), *state[2:]))
            torch.set_rng_state(rng["torch"])
            if device.type == "cuda" and "cuda" in rng:
                torch.cuda.set_rng_state_all(rng["cuda"])
    if start >= args.epochs:
        raise ValueError("Checkpoint has already completed the requested epoch schedule")
    run = Path(args.output_dir) / (args.run_name or datetime.now().strftime("coherefusion_%Y%m%d_%H%M%S"))
    weights = run / "weights"
    weights.mkdir(parents=True, exist_ok=False)
    (run / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    (run / "split_manifest.json").write_bytes(Path(args.split_manifest).read_bytes())
    print(f"Architecture: {ARCHITECTURE}; frozen CLIP + DeepLab; fusion parameters initialized randomly")
    writer = SummaryWriter(run / "tensorboard")
    try:
        for epoch in range(start, args.epochs):
            stats = epoch_pass(model, train_loader, objective, device, epoch, optimizer, scheduler,
                               scaler=scaler, grad_clip=args.grad_clip)
            for key, value in stats.items():
                writer.add_scalar(f"train/{key}", value, epoch)
            if (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs:
                stats = epoch_pass(model, val_loader, objective, device, epoch)
                selection = stats["quality_worst"] + .1 * stats["quality_mean"]
                improved = selection < best
                best = min(best, selection)
                for key, value in stats.items():
                    writer.add_scalar(f"val/{key}", value, epoch)
                save_checkpoint(weights / "last.pth", model, optimizer, scheduler, epoch, best, args, scaler)
                if improved:
                    save_checkpoint(weights / "best.pth", model, optimizer, scheduler, epoch, best, args, scaler)
                print(f"val selection={selection:.6f}; best={best:.6f}")
    finally:
        writer.close()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split-manifest", required=True, help="Disjoint train/val/test scene IDs in JSON")
    parser.add_argument("--output-dir", default="./experiments_cohere")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--device", default="cuda")
    integers = {"epochs": 40, "batch_size": 8, "crop_size": 128, "dim": 32, "global_top_k": 4,
                "warmup_epochs": 3, "bins": 64, "val_every": 1, "workers": 4, "seed": 3407}
    numbers = {"branch_lr": 2e-5, "router_lr": 1e-4, "weight_decay": 1e-4, "grad_clip": 1.,
               "entropy_margin": .12, "mi_floor": 2.8, "scd_floor": 1.55, "scd_weight": 1.5,
               "text_weight": .2, "dynamic_weight": 3., "dynamic_margin": .001,
               "counterfactual_weight": .3, "counterfactual_margin": .025, "residual_weight": .02,
               "global_weight": .35, "semantic_weight": .35, "fixed_weight": 6.,
               "fixed_margin": .001, "diversity_weight": .05, "diversity_margin": .005,
               "dynamic_rho": .65, "fixed_rho": .75}
    for key, value in integers.items():
        parser.add_argument("--" + key.replace("_", "-"), type=int, default=value)
    for key, value in numbers.items():
        parser.add_argument("--" + key.replace("_", "-"), type=float, default=value)
    parser.add_argument("--fixed-targets", type=float, nargs=8, default=None)
    parser.add_argument("--fixed-importance", type=float, nargs=8, default=None)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
