"""Evaluate registered fusion outputs with source and optional clean references."""

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from data.io import image_index, paired_images
from data.tasks import TASKS
from scripts.metrics import FusionMetrics


KEYS = ("EN", "SD", "SF", "AG", "MI", "SSIM", "SCD", "Qabf", "VIF")


def read(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)


def one(item):
    infrared, visible, fused = map(read, item)
    if infrared.shape != fused.shape or visible.shape != fused.shape:
        raise ValueError(f"Registered evaluation images have mismatched dimensions: {item}")
    dx, dy = fused[:-1, 1:] - fused[:-1, :-1], fused[1:, :-1] - fused[:-1, :-1]
    values = (FusionMetrics.en(fused), FusionMetrics.sd(fused), FusionMetrics.sf(fused),
              np.sqrt((dx * dx + dy * dy) / 2).mean(),
              FusionMetrics.mi(fused, infrared, visible) / math.log(2),
              FusionMetrics.ssim_sum(fused, infrared, visible) / 2,
              FusionMetrics.scd(fused, infrared, visible), FusionMetrics.qabf(fused, infrared, visible),
              FusionMetrics.vif(fused, infrared, visible))
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite evaluation metric for {item[-1]}")
    return values


def main(args):
    if args.workers < 1:
        raise ValueError("Workers must be positive")
    source, outputs = Path(args.source), Path(args.fused)
    clean_suffix = "_gt" if args.clean_reference else ""
    if all((source / task).is_dir() for task in TASKS):
        groups = [(task, source / task / "test" / ("Visible" + clean_suffix),
                   source / task / "test" / ("Infrared" + clean_suffix), outputs / task) for task in TASKS]
    else:
        groups = []
        for root in (source, source / "test"):
            for visible, infrared in (("Visible" + clean_suffix, "Infrared" + clean_suffix),
                                      ("vi", "ir") if not args.clean_reference else ("vi_gt", "ir_gt")):
                if (root / visible).is_dir() and (root / infrared).is_dir():
                    groups = [("dataset", root / visible, root / infrared, outputs)]
                    break
            if groups:
                break
        if not groups:
            raise ValueError(f"Reference image directories not found below {source}")
    records, arrays = {}, []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for task, visible_dir, infrared_dir, fused_dir in groups:
            pairs = paired_images(visible_dir, infrared_dir)
            fused = image_index(fused_dir)
            if set(fused) != {visible.stem for visible, _ in pairs}:
                raise ValueError(f"Fused output set does not match the reference pairs for {task}")
            items = [(infrared, visible, fused[visible.stem]) for visible, infrared in pairs]
            values = np.asarray(list(pool.map(one, items)), dtype=np.float64)
            arrays.append(values)
            records[task] = {"count": len(values), "metrics": dict(zip(KEYS, values.mean(0).tolist())),
                             "std": dict(zip(KEYS, values.std(0).tolist()))}
    values = np.concatenate(arrays)
    macro = np.stack([array.mean(0) for array in arrays]).mean(0)
    result = {"count": len(values), "metrics": dict(zip(KEYS, values.mean(0).tolist())),
              "macro_metrics": dict(zip(KEYS, macro.tolist())), "by_task": records,
              "reference": "clean" if args.clean_reference else "input"}
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--fused", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--clean-reference", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
