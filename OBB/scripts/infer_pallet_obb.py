from __future__ import annotations

import argparse
import json
from pathlib import Path


def run(args: argparse.Namespace) -> None:
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    results = model.predict(
        source=args.source,
        task="obb",
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        max_det=args.max_det,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        project=str(out_dir.resolve()),
        name=args.run_name,
        exist_ok=True,
        stream=False,
        verbose=False,
    )

    total_frames = len(results)
    total_obb = 0
    for result in results:
        if hasattr(result, "obb") and result.obb is not None:
            total_obb += len(result.obb)

    summary = {
        "model": str(model_path.resolve()),
        "source": args.source,
        "output_dir": str((out_dir / args.run_name).resolve()),
        "frames_or_images": total_frames,
        "pred_obb": total_obb,
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
    }
    summary_path = out_dir / args.run_name / "inference_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[DONE] Saved OBB predictions to: {out_dir / args.run_name}")
    print(f"[DONE] Inference summary: {summary_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run OBB pallet inference with a YOLO OBB model.")
    p.add_argument("--model", required=True, help="Path to trained .pt OBB model.")
    p.add_argument("--source", required=True, help="Image/video/folder path.")
    p.add_argument("--output-dir", default="II/OBB/reports/roboflo2_pallet_obb_v1/inference")
    p.add_argument("--run-name", default="predict")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.50)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-det", type=int, default=50)
    p.add_argument("--save-txt", action="store_true")
    p.add_argument("--save-conf", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
