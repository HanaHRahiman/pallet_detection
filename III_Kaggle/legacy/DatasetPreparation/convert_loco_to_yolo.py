"""
Convert LOCO COCO format to pallet-only YOLO format.

This script:
1. Finds pallet category IDs in LOCO COCO JSON
2. Converts only pallet annotations to YOLO format
3. Copies images and creates label files
"""

import os
import json
import shutil
from pathlib import Path


def find_pallet_categories(coco_data):
    """Find all category IDs that contain 'pallet' in the name."""
    pallet_cat_ids = {
        c["id"] for c in coco_data["categories"] 
        if "pallet" in c["name"].lower()
    }
    return pallet_cat_ids


def coco_box_to_yolo(bbox, img_width, img_height):
    """
    Convert COCO bbox [x, y, width, height] to YOLO format [xc, yc, width, height] (normalized).
    
    COCO: top-left corner (x, y) + width, height
    YOLO: center (xc, yc) + width, height (all normalized 0-1)
    """
    x, y, bw, bh = bbox
    
    # Calculate center coordinates
    xc = (x + bw / 2) / img_width
    yc = (y + bh / 2) / img_height
    
    # Normalize width and height
    bw_norm = bw / img_width
    bh_norm = bh / img_height
    
    return xc, yc, bw_norm, bh_norm


def convert_loco_split(
    ann_json_path,
    img_dir,
    out_root,
    split_name,
    pallet_cat_ids
):
    """
    Convert one split (train/val/test) of LOCO to YOLO format.
    
    Args:
        ann_json_path: Path to COCO JSON annotation file
        img_dir: Directory containing images
        out_root: Output root directory (datasets/pallet_v1)
        split_name: 'train', 'val', or 'test'
        pallet_cat_ids: Set of category IDs to keep (pallets only)
    """
    print(f"\n=== Processing {split_name} split ===")
    
    # Load COCO JSON
    if not os.path.exists(ann_json_path):
        print(f"  WARNING: {ann_json_path} not found, skipping {split_name}")
        return 0
    
    with open(ann_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Build image lookup
    imgs = {im["id"]: im for im in data["images"]}
    
    # Group annotations by image_id, filtering for pallets only
    anns_by_img = {}
    for a in data["annotations"]:
        if a["category_id"] in pallet_cat_ids:
            anns_by_img.setdefault(a["image_id"], []).append(a)
    
    # Create output directories
    out_img_dir = Path(out_root) / "images" / split_name
    out_label_dir = Path(out_root) / "labels" / split_name
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)
    
    img_dir = Path(img_dir)
    converted_count = 0
    skipped_count = 0
    
    # Process each image that has pallet annotations
    for image_id, ann_list in anns_by_img.items():
        im = imgs[image_id]
        w, h = im["width"], im["height"]
        file_name = im["file_name"]
        
        # Source image path
        src_img = img_dir / file_name
        
        # Skip if image doesn't exist
        if not src_img.exists():
            print(f"  WARNING: Image not found: {src_img}")
            skipped_count += 1
            continue
        
        # Destination paths
        dst_img = out_img_dir / Path(file_name).name
        label_path = out_label_dir / (dst_img.stem + ".txt")
        
        # Copy image
        shutil.copy2(src_img, dst_img)
        
        # Write YOLO label file (class 0 = pallet)
        with open(label_path, "w") as f:
            for a in ann_list:
                xc, yc, bw, bh = coco_box_to_yolo(a["bbox"], w, h)
                # Validate normalized coordinates
                if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 <= bw <= 1 and 0 <= bh <= 1):
                    print(f"  WARNING: Invalid bbox in {label_path}: xc={xc}, yc={yc}, bw={bw}, bh={bh}")
                f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
        
        converted_count += 1
    
    print(f"  Converted: {converted_count} images")
    if skipped_count > 0:
        print(f"  Skipped: {skipped_count} images (not found)")
    
    return converted_count


def main():
    """Main conversion function."""
    # Paths - adjust these to match your LOCO structure
    loco_root = Path("datasets/raw/LOCO")
    out_root = Path("datasets/pallet_v1")
    
    # Expected LOCO structure (adjust if different):
    # LOCO/
    #   annotations/
    #     instances_train.json
    #     instances_val.json
    #     instances_test.json
    #   images/
    #     train/
    #     val/
    #     test/
    
    # Find pallet categories from train JSON
    train_ann_path = loco_root / "annotations" / "instances_train.json"
    
    if not train_ann_path.exists():
        print(f"ERROR: LOCO train annotations not found at {train_ann_path}")
        print("\nPlease ensure LOCO dataset is placed in: datasets/raw/LOCO/")
        print("Expected structure:")
        print("  LOCO/")
        print("    annotations/")
        print("      instances_train.json")
        print("      instances_val.json")
        print("      instances_test.json")
        print("    images/")
        print("      train/")
        print("      val/")
        print("      test/")
        return
    
    # Load and find pallet categories
    with open(train_ann_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    
    pallet_cat_ids = find_pallet_categories(train_data)
    
    if not pallet_cat_ids:
        print("ERROR: No pallet category found in LOCO dataset!")
        print("\nAvailable categories:")
        for c in train_data["categories"]:
            print(f"  ID {c['id']}: {c['name']}")
        return
    
    print(f"Found pallet category IDs: {pallet_cat_ids}")
    for cat_id in pallet_cat_ids:
        cat_name = next(c["name"] for c in train_data["categories"] if c["id"] == cat_id)
        print(f"  ID {cat_id}: {cat_name}")
    
    # Convert each split
    splits = [
        ("train", "instances_train.json", "train"),
        ("val", "instances_val.json", "val"),
        ("test", "instances_test.json", "test"),
    ]
    
    total_converted = 0
    for split_name, json_name, img_subdir in splits:
        ann_path = loco_root / "annotations" / json_name
        img_dir = loco_root / "images" / img_subdir
        
        count = convert_loco_split(
            ann_path,
            img_dir,
            out_root,
            split_name,
            pallet_cat_ids
        )
        total_converted += count
    
    print(f"\n=== Conversion Complete ===")
    print(f"Total images converted: {total_converted}")


if __name__ == "__main__":
    main()
