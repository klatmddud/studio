from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.runtime.config import load_runtime_config, resolve_device
from scripts.runtime.data import build_train_dataloaders
from scripts.runtime.engine import fit, seed_everything
from scripts.runtime.registry import build_model_from_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a detection model from YAML configs.")
    parser.add_argument("--config", required=True, help="Path to the train runtime YAML.")
    parser.add_argument("--model", required=True, help="Path to the model YAML.")
    parser.add_argument(
        "--data",
        default=None,
        help=(
            "Optional dataset selector for runtime data env vars "
            "(for example: kitti, bdd100k, bdd10k)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional override for runtime.output_dir.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional override for runtime.device (for example: cpu, cuda, auto).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_config, runtime_config_path = load_runtime_config(
        args.config,
        mode="train",
        dataset=args.data,
    )
    if args.output_dir:
        runtime_config["output_dir"] = str(Path(args.output_dir).expanduser().resolve())
        runtime_config["checkpoint"]["dir"] = str(
            (Path(runtime_config["output_dir"]) / "checkpoints").resolve()
        )
    if args.device:
        runtime_config["device"] = args.device

    seed_everything(runtime_config["seed"])
    model, model_config, arch, model_config_path = build_model_from_path(
        args.model,
        runtime_config=runtime_config,
    )
    device = resolve_device(runtime_config["device"])
    train_loader, val_loader = build_train_dataloaders(runtime_config, arch=arch)

    print(f"Starting training: arch={arch} device={device.type}")
    print(f"train_config={runtime_config_path}")
    print(f"model_config={model_config_path}")
    if runtime_config.get("_dataset"):
        print(f"dataset={runtime_config['_dataset']}")

    fit(
        model=model,
        model_config=model_config,
        model_config_path=model_config_path,
        runtime_config=runtime_config,
        runtime_config_path=runtime_config_path,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        arch=arch,
    )


if __name__ == "__main__":
    main()
