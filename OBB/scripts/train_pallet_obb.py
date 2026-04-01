from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


DEFAULT_CONFIG = Path("II/OBB/configs/train_obb_yolo11m.yaml")
DEFAULT_DATA = Path("II/OBB/data/roboflo2_pallet_obb_v3/data.yaml")
DEFAULT_RUN_META = Path("II/OBB/reports/roboflo2_pallet_obb_v3/train_run.json")


def run(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    model_name = cfg.pop("model", "yolo11m-obb.pt")
    data_yaml = Path(cfg.pop("data", str(args.data)))
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_yaml}")

    if args.fast and model_name == "yolo11m-obb.pt":
        model_name = "yolo11s-obb.pt"

    cfg["data"] = str(data_yaml.resolve())
    cfg.setdefault("project", str(Path("II/OBB/runs/obb").resolve()))
    cfg.setdefault("name", "roboflo2_pallet_obb")
    cfg.setdefault("task", "obb")

    run_meta = {
        "model": model_name,
        "config": cfg,
        "data_yaml": str(data_yaml.resolve()),
        "mode": "dry-run" if args.dry_run else "train",
    }
    Path(args.run_meta).parent.mkdir(parents=True, exist_ok=True)
    Path(args.run_meta).write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    print(f"[INFO] Run config written: {args.run_meta}")

    if args.dry_run:
        print("[DRY-RUN] Skipping training execution.")
        return

    from ultralytics import YOLO

    model = YOLO(model_name)
    print(f"[INFO] Training OBB model: {model_name}")
    model.train(**cfg)
    print("[DONE] OBB training completed.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train one-class pallet OBB model.")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="Training YAML config.")
    p.add_argument("--data", default=str(DEFAULT_DATA), help="Path to cleaned OBB data.yaml.")
    p.add_argument("--run-meta", default=str(DEFAULT_RUN_META), help="Output JSON of resolved run config.")
    p.add_argument("--fast", action="store_true", help="Use yolo11s-obb fallback for faster iteration.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
