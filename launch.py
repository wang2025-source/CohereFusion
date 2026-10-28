"""CohereFusion training, inference and evaluation entry point."""
import argparse
import json
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("mode", choices=("train", "infer", "evaluate"))
    parser.add_argument("--config", help="Training JSON defaults; explicit CLI flags override them")
    if len(sys.argv) == 1 or sys.argv[1:] in (["-h"], ["--help"]):
        parser.print_help()
        return
    args, remaining = parser.parse_known_args()
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    if args.mode != "evaluate" and not any(flag in remaining for flag in ("-h", "--help")):
        import clip
        import torch
        original_load = clip.load

        def local_load(name, *positional, **kwargs):
            kwargs.setdefault("download_root", str(root / "assets" / "clip"))
            return original_load(name, *positional, **kwargs)

        clip.load = local_load
        torch.hub.set_dir(str(root / "assets" / "torch"))
    defaults = []
    if args.config:
        if args.mode != "train":
            parser.error("--config is supported only for train")
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            parser.error("Training config must be a JSON object")
        for key, value in config.items():
            if value is None or value is False:
                continue
            defaults.append("--" + key.replace("_", "-"))
            if value is True:
                continue
            if isinstance(value, list):
                defaults.extend(str(item) for item in value)
            else:
                defaults.append(str(value))
    entry = {"train": "train.py", "infer": "infer.py", "evaluate": "evaluate.py"}[args.mode]
    sys.argv = [str(root / entry), *defaults, *remaining]
    runpy.run_path(str(root / entry), run_name="__main__")


if __name__ == "__main__":
    main()
