from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml 


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect_images(path: Path) -> list[Path]:
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _clip01(v: float) -> float:
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _shoelace_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        area += (x1 * y2) - (x2 * y1)
    return abs(area) * 0.5


def _order_quad(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    start = np.lexsort((ordered[:, 0], ordered[:, 1]))[0]
    return np.roll(ordered, -start, axis=0)


def _polygon_to_obb(values: list[float], stats: Counter) -> list[float] | None:
    if len(values) < 6 or len(values) % 2 != 0:
        stats["dropped_malformed_shape"] += 1
        return None

    points = np.array(list(zip(values[0::2], values[1::2])), dtype=np.float32)
    if _shoelace_area(points) <= 1e-6:
        stats["dropped_zero_area_polygon"] += 1
        return None

    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    box = np.array([[_clip01(float(x)), _clip01(float(y))] for x, y in box], dtype=np.float32)
    box = _order_quad(box)

    if _shoelace_area(box) <= 1e-6:
        stats["dropped_zero_area_obb"] += 1
        return None

    stats["objects_converted"] += 1
    return box.reshape(-1).tolist()


def _box_to_polygon(values: list[float]) -> list[float]:
    """Convert YOLO xywh box to 4-point polygon (clockwise)."""
    if len(values) != 4:
        return values
    xc, yc, w, h = values
    x1, y1 = xc - w * 0.5, yc - h * 0.5
    x2, y2 = xc + w * 0.5, yc + h * 0.5
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def _format_obb_line(cls_id: int, coords: list[float]) -> str:
    return f"{cls_id} " + " ".join(f"{v:.6f}" for v in coords)


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _save_visual_samples(dataset_root: Path, report_dir: Path, sample_count: int) -> None:
    sample_dir = report_dir / "visual_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    all_images: list[tuple[str, Path]] = []
    for split in ("train", "valid", "test"):
        for img_path in _collect_images(dataset_root / split / "images"):
            all_images.append((split, img_path))

    for idx, (split, img_path) in enumerate(all_images[:sample_count], start=1):
        label_path = dataset_root / split / "labels" / f"{img_path.stem}.txt"
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        if label_path.exists():
            for raw_line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = raw_line.strip().split()
                if len(parts) != 9:
                    continue
                coords = [float(v) for v in parts[1:]]
                pts = np.array(
                    [[int(coords[i] * w), int(coords[i + 1] * h)] for i in range(0, len(coords), 2)],
                    dtype=np.int32,
                ).reshape((-1, 1, 2))
                cv2.polylines(image, [pts], True, (0, 255, 255), 2)
        out_name = f"sample_{idx:03d}_{split}_{img_path.name}"
        cv2.imwrite(str(sample_dir / out_name), image)


def _parse_keep_classes(s: str) -> list[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return sorted({int(float(p)) for p in parts})


def run(args: argparse.Namespace) -> None:
    src = Path(args.src).resolve()
    out = Path(args.out).resolve()
    report_dir = Path(args.report_dir).resolve()

    if not src.exists():
        raise FileNotFoundError(f"Source dataset not found: {src}")
    if not (src / "data.yaml").exists():
        raise FileNotFoundError(f"Source data.yaml not found: {src / 'data.yaml'}")

    if out.exists() and args.reset_out:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    with open(src / "data.yaml", "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f) or {}

    keep_classes = _parse_keep_classes(args.keep_classes)
    if not keep_classes:
        raise ValueError("keep_classes produced an empty set.")

    source_names: list[str] = data_cfg.get("names", [])
    if not isinstance(source_names, list):
        source_names = []

    # build mapping from source class id -> target class id
    if args.single_class:
        id_map = {cid: 0 for cid in keep_classes}
        target_names = ["pallet"]
    else:
        target_names = []
        id_map = {}
        for new_id, cid in enumerate(keep_classes):
            id_map[cid] = new_id
            name = source_names[cid] if cid < len(source_names) else f"class_{cid}"
            target_names.append(name)

    stats = Counter()
    split_stats: dict[str, Counter] = {split: Counter() for split in ("train", "valid", "test")}

    for split in ("train", "valid", "test"):
        src_img_dir = src / split / "images"
        src_lbl_dir = src / split / "labels"
        dst_img_dir = out / split / "images"
        dst_lbl_dir = out / split / "labels"
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        images = _collect_images(src_img_dir)
        split_stats[split]["images"] = len(images)

        for img_path in images:
            shutil.copy2(img_path, dst_img_dir / img_path.name)
            src_label = src_lbl_dir / f"{img_path.stem}.txt"
            dst_label = dst_lbl_dir / f"{img_path.stem}.txt"

            out_lines: list[str] = []
            if src_label.exists():
                for raw_line in src_label.read_text(encoding="utf-8", errors="ignore").splitlines():
                    parts = raw_line.strip().split()
                    if len(parts) < 5:  # cls + at least box (xywh) or polygon
                        stats["dropped_parse_error"] += 1
                        split_stats[split]["dropped_parse_error"] += 1
                        continue
                    try:
                        cls_id = int(float(parts[0]))
                        values = [float(v) for v in parts[1:]]
                    except Exception:
                        stats["dropped_parse_error"] += 1
                        split_stats[split]["dropped_parse_error"] += 1
                        continue

                    split_stats[split]["objects_raw"] += 1
                    stats["objects_raw"] += 1
                    if cls_id not in keep_classes:
                        stats["dropped_non_whitelist"] += 1
                        split_stats[split]["dropped_non_whitelist"] += 1
                        continue

                    if len(values) == 4:
                        values = _box_to_polygon(values)
                        stats["converted_box_to_polygon"] += 1
                        split_stats[split]["converted_box_to_polygon"] += 1

                    obb_coords = _polygon_to_obb(values, stats)
                    if obb_coords is None:
                        split_stats[split]["dropped_invalid_geometry"] += 1
                        continue

                    out_lines.append(_format_obb_line(id_map[cls_id], obb_coords))
                    split_stats[split]["objects_kept"] += 1
                    stats["objects_kept"] += 1
            else:
                split_stats[split]["missing_label"] += 1

            text = "\n".join(out_lines)
            dst_label.write_text(text + ("\n" if text else ""), encoding="utf-8")

    out_data = {
        "path": str(out),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(target_names),
        "names": target_names,
    }
    with open(out / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(out_data, f, sort_keys=False)

    report = {
        "source_dataset": str(src),
        "output_dataset": str(out),
        "source_task": "segment",
        "output_task": "obb",
        "keep_classes": keep_classes,
        "id_map": id_map,
        "target_names": target_names,
        "objects_raw": int(stats.get("objects_raw", 0)),
        "objects_kept": int(stats.get("objects_kept", 0)),
        "global_stats": {k: int(v) for k, v in sorted(stats.items())},
        "split_stats": {split: {k: int(v) for k, v in sorted(counter.items())} for split, counter in split_stats.items()},
    }
    _save_json(report_dir / "obb_prep_report.json", report)
    _save_visual_samples(out, report_dir, sample_count=args.visual_samples)

    print(f"[DONE] OBB dataset written to: {out}")
    print(f"[DONE] OBB data.yaml: {out / 'data.yaml'}")
    print(f"[DONE] OBB report: {report_dir / 'obb_prep_report.json'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert polygon labels into YOLO OBB labels with class filtering/mapping.")
    p.add_argument("--src", default="II/data/raw/roboflo2", help="Source segmentation dataset.")
    p.add_argument("--out", default="II/OBB/data/roboflo2_pallet_obb_v3", help="Output OBB dataset root.")
    p.add_argument("--report-dir", default="II/OBB/reports/roboflo2_pallet_obb_v3", help="Output report directory.")
    p.add_argument(
        "--keep-classes",
        default="0,1,2,4,5,6,7",
        help="Comma-separated source class ids to keep (e.g., '0,1,4,5').",
    )
    p.add_argument("--single-class", action="store_true", default=True, help="Collapse kept classes to a single class 0.")
    p.add_argument("--multi-class", dest="single_class", action="store_false", help="Keep distinct classes (contiguous remap).")
    p.add_argument("--visual-samples", type=int, default=40, help="Number of OBB visual samples to render.")
    p.add_argument("--reset-out", action="store_true", help="Delete the output directory before writing.")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
