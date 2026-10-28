"""Registered image-pair and text-description loading."""

import json
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def image_index(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    result = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            if path.stem in result:
                raise ValueError(f"Ambiguous image stem {path.stem!r} in {directory}")
            result[path.stem] = path
    if not result:
        raise ValueError(f"No supported images in {directory}")
    return result


def paired_images(visible_dir, infrared_dir):
    visible, infrared = image_index(visible_dir), image_index(infrared_dir)
    if visible.keys() != infrared.keys():
        missing_ir = sorted(visible.keys() - infrared.keys())[:5]
        missing_vi = sorted(infrared.keys() - visible.keys())[:5]
        raise ValueError(f"Unpaired inputs: missing infrared={missing_ir}, missing visible={missing_vi}")
    return [(visible[stem], infrared[stem]) for stem in sorted(visible)]


def descriptions(path):
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict) or not all(
            isinstance(key, str) and isinstance(value, str) and value.strip()
            for key, value in values.items()):
        raise ValueError("Descriptions must be a JSON object mapping image names to nonempty text")
    return {key: value.strip() for key, value in values.items()}


def image_description(values, task, image, default):
    for key in (f"{task}/{image.name}", f"{task}/{image.stem}", image.name, image.stem):
        if key in values:
            return values[key]
    if values and default is None:
        raise ValueError(f"Description missing for {task}/{image.name}")
    return default
