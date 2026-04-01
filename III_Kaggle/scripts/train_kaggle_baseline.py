from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = EXPERIMENT_ROOT.parent


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def save_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _count_files(path: Path) -> int:
    return sum(1 for item in path.iterdir() if item.is_file())


def _resolve_split_dir(source_root: Path, raw_value: str, fallback_split: str) -> Path:
    candidates = [
        (source_root / raw_value.replace("../", "")).resolve(),
        (source_root / fallback_split / "images").resolve(),
        (source_root / raw_value).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _build_resolved_data_yaml(source_yaml: Path, out_yaml: Path) -> dict:
    src = load_yaml(source_yaml)
    source_root = source_yaml.parent.resolve()

    payload = {
        "train": str(_resolve_split_dir(source_root, src["train"], "train")),
        "val": str(_resolve_split_dir(source_root, src["val"], "valid")),
        "test": str(_resolve_split_dir(source_root, src["test"], "test")),
        "nc": 1,
        "names": ["pallet"],
    }
    save_yaml(out_yaml, payload)
    return payload


def run(args: argparse.Namespace) -> None:
    cfg = load_yaml(_resolve_repo_path(args.config))
    source_yaml = _resolve_repo_path(cfg.pop("data_source"))
    resolved_data_yaml = _resolve_repo_path(args.resolved_data_out)
    run_meta_out = _resolve_repo_path(args.run_meta_out)

    resolved_payload = _build_resolved_data_yaml(source_yaml, resolved_data_yaml)

    model_name = cfg.pop("model", "yolov8s.pt")
    device = cfg.get("device", "auto")
    if device == "auto":
        cfg["device"] = 0 if torch.cuda.is_available() else "cpu"

    cfg["data"] = str(resolved_data_yaml.resolve())

    project = Path(cfg.get("project", EXPERIMENT_ROOT / "runs" / "detect"))
    if not project.is_absolute():
        project = (PROJECT_ROOT / project).resolve()
    cfg["project"] = str(project)

    run_meta = {
        "generated_at": utc_now_iso(),
        "model": model_name,
        "source_yaml": str(source_yaml),
        "resolved_data_yaml": str(resolved_data_yaml.resolve()),
        "device": cfg["device"],
        "train_images": _count_files(Path(resolved_payload["train"])),
        "val_images": _count_files(Path(resolved_payload["val"])),
        "test_images": _count_files(Path(resolved_payload["test"])),
        "config": cfg,
    }
    save_json(run_meta_out, run_meta)
    print(f"[INFO] Run metadata written: {run_meta_out}")

    if args.dry_run:
        print("[DRY-RUN] Skipping YOLO training execution.")
        return

    from ultralytics import YOLO

    model = YOLO(model_name)
    print(f"[INFO] Training detection baseline with model={model_name}")
    model.train(**cfg)
    print("[DONE] Kaggle baseline training completed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the old Kaggle-style YOLO detection baseline.")
    parser.add_argument("--config", default=str(EXPERIMENT_ROOT / "configs" / "train_detect_kaggle_yolov8s.yaml"))
    parser.add_argument(
        "--resolved-data-out",
        default="III_kaggle/reports/kaggle_baseline/kaggle_data_resolved.yaml",
    )
    parser.add_argument(
        "--run-meta-out",
        default="III_kaggle/reports/kaggle_baseline/train_run.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
