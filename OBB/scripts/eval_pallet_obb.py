from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def _pick(metrics: dict, key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def run(args: argparse.Namespace) -> None:
    model_path = Path(args.model)
    data_path = Path(args.data)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data yaml not found: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f) or {}

    from ultralytics import YOLO

    model = YOLO(str(model_path))

    val_results = model.val(
        data=str(data_path.resolve()),
        split="val",
        task="obb",
        conf=args.conf,
        iou=args.iou,
        project=str(Path(args.project).resolve()),
        name=args.run_name + "_val",
        save_json=args.save_json,
    )

    test_results = model.val(
        data=str(data_path.resolve()),
        split="test",
        task="obb",
        conf=args.conf,
        iou=args.iou,
        project=str(Path(args.project).resolve()),
        name=args.run_name + "_test",
        save_json=args.save_json,
    )

    val_metrics = val_results.results_dict if hasattr(val_results, "results_dict") else {}
    test_metrics = test_results.results_dict if hasattr(test_results, "results_dict") else {}

    summary = {
        "model": str(model_path.resolve()),
        "data": str(data_path.resolve()),
        "dataset_root": str(Path(data_cfg.get("path", "")).resolve()) if data_cfg.get("path") else None,
        "conf": args.conf,
        "iou": args.iou,
        "val": {
            "precision": _pick(val_metrics, "metrics/precision(B)"),
            "recall": _pick(val_metrics, "metrics/recall(B)"),
            "map50": _pick(val_metrics, "metrics/mAP50(B)"),
            "map50_95": _pick(val_metrics, "metrics/mAP50-95(B)"),
        },
        "test": {
            "precision": _pick(test_metrics, "metrics/precision(B)"),
            "recall": _pick(test_metrics, "metrics/recall(B)"),
            "map50": _pick(test_metrics, "metrics/mAP50(B)"),
            "map50_95": _pick(test_metrics, "metrics/mAP50-95(B)"),
        },
        "raw_metrics": {"val": val_metrics, "test": test_metrics},
    }

    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "# OBB Evaluation",
        "",
        f"- Model: `{summary['model']}`",
        f"- Data: `{summary['data']}`",
        f"- conf={args.conf}, iou={args.iou}",
        "",
        "## Validation",
        f"- Precision: {summary['val']['precision']}",
        f"- Recall: {summary['val']['recall']}",
        f"- mAP50: {summary['val']['map50']}",
        f"- mAP50-95: {summary['val']['map50_95']}",
        "",
        "## Test",
        f"- Precision: {summary['test']['precision']}",
        f"- Recall: {summary['test']['recall']}",
        f"- mAP50: {summary['test']['map50']}",
        f"- mAP50-95: {summary['test']['map50_95']}",
        "",
    ]
    summary_md = Path(args.summary_md)
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    summary_md.write_text("\n".join(md), encoding="utf-8")

    print(f"[DONE] Evaluation summary JSON: {summary_json}")
    print(f"[DONE] Evaluation summary MD: {summary_md}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate pallet OBB model.")
    p.add_argument("--model", required=True, help="Path to trained OBB .pt")
    p.add_argument("--data", default="II/OBB/data/roboflo2_pallet_obb_v1/data.yaml")
    p.add_argument("--project", default="II/OBB/runs/obb_eval")
    p.add_argument("--run-name", default="roboflo2_obb_eval")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.50)
    p.add_argument("--save-json", action="store_true")
    p.add_argument("--summary-json", default="II/OBB/reports/roboflo2_pallet_obb_v1/eval_summary.json")
    p.add_argument("--summary-md", default="II/OBB/reports/roboflo2_pallet_obb_v1/eval_summary.md")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
