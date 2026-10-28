"""Task-balanced paired data with explicit, scene-disjoint partitions."""

import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from data.io import image_index, descriptions, image_description
from data.tasks import TASKS, TASK_KEYS, DEFAULT_PROMPTS


def load_scene_splits(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Split manifest must be a JSON object")
    partitions = {}
    for split in ("train", "val", "test"):
        items = manifest.get(split)
        if not isinstance(items, list) or not items or not all(isinstance(x, str) and x.strip() for x in items):
            raise ValueError(f"Split manifest needs a nonempty list of scene IDs for {split}")
        partitions[split] = {Path(item).stem for item in items}
        if len(partitions[split]) != len(items):
            raise ValueError(f"Duplicate scene IDs in {split} split")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = partitions[a] & partitions[b]
        if overlap:
            raise ValueError(f"Scene leakage between {a} and {b}: {sorted(overlap)[:5]}")
    return partitions


class CohereEMSDataset(Dataset):
    def __init__(self, root, split, manifest_path, crop_size=128, seed=3407):
        if split not in ("train", "val"):
            raise ValueError("Training loader accepts train or val only")
        if crop_size < 8 or crop_size % 8:
            raise ValueError("Crop size must be a positive multiple of 8")
        partitions = load_scene_splits(manifest_path)
        self.root, self.split, self.crop_size = Path(root), split, crop_size
        self.seed, self.epoch, self.records = seed, 0, {}
        self.task_prompts, self.image_prompts = {}, {}
        found = set()
        for directory in TASKS:
            task = TASK_KEYS[directory]
            base = self.root / directory / "train"
            folders = [image_index(base / name) for name in
                       ("Visible", "Infrared", "Visible_gt", "Infrared_gt")]
            scene_ids = sorted(set(folders[0]) & partitions[split])
            self.records[task] = []
            for scene_id in scene_ids:
                if any(scene_id not in folder for folder in folders):
                    raise ValueError(f"Incomplete registered training pair: {directory}/{scene_id}")
                self.records[task].append(tuple(folder[scene_id] for folder in folders))
            found.update(scene_ids)
            text_file = base / "text.txt"
            self.task_prompts[task] = ([line.strip() for line in text_file.read_text(encoding="utf-8").splitlines()
                                       if line.strip()] if text_file.exists() else []) or [DEFAULT_PROMPTS[task]]
            self.image_prompts[task] = descriptions(base / "text.json") if (base / "text.json").exists() else {}
        if missing := partitions[split] - found:
            raise ValueError(f"Manifest {split} scenes absent from dataset: {sorted(missing)[:5]}")
        self.tasks = tuple(self.records)
        if any(not self.records[task] for task in self.tasks):
            raise ValueError(f"Empty task in {split} split: {self.counts}")
        self.samples_per_task = max(map(len, self.records.values())) if split == "train" else None
        self.index = ([(task, i) for task in self.tasks for i in range(len(self.records[task]))]
                      if split == "val" else None)

    @property
    def counts(self):
        return {task: len(records) for task, records in self.records.items()}

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.index) if self.index is not None else self.samples_per_task * len(self.tasks)

    def __getitem__(self, item):
        if self.split == "val":
            task, sample_index = self.index[item]
            rng = random.Random(self.seed + item)
        else:
            task = self.tasks[item % len(self.tasks)]
            sample_index = (item // len(self.tasks)) % len(self.records[task])
            rng = random.Random(self.seed + self.epoch * len(self) + item)
        paths = self.records[task][sample_index]
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        if len({image.size for image in images}) != 1:
            raise ValueError(f"Mismatched registered image sizes for {paths[0]}")
        images = self._augment(images, rng) if self.split == "train" else self._pad_multiple(images)
        values = self.image_prompts[task]
        prompt = image_description(values, task, paths[0], None) if values else rng.choice(self.task_prompts[task])
        return *(TF.to_tensor(image) for image in images), task, prompt, paths[0].stem

    def _augment(self, images, rng):
        width, height = images[0].size
        pad = (0, 0, max(0, self.crop_size - width), max(0, self.crop_size - height))
        if any(pad):
            images = [TF.pad(image, pad, padding_mode="edge") for image in images]
        width, height = images[0].size
        left, top = rng.randint(0, width - self.crop_size), rng.randint(0, height - self.crop_size)
        images = [TF.crop(image, top, left, self.crop_size, self.crop_size) for image in images]
        if rng.random() < .5:
            images = [TF.hflip(image) for image in images]
        if rng.random() < .5:
            images = [TF.vflip(image) for image in images]
        return images

    @staticmethod
    def _pad_multiple(images):
        width, height = images[0].size
        pad = (0, 0, (-width) % 8, (-height) % 8)
        return [TF.pad(image, pad, padding_mode="edge") for image in images] if any(pad) else images

    @staticmethod
    def collate_fn(batch):
        visible, infrared, visible_gt, infrared_gt, tasks, prompts, names = zip(*batch)
        return (torch.stack(visible), torch.stack(infrared), torch.stack(visible_gt),
                torch.stack(infrared_gt), tasks, prompts, names)
