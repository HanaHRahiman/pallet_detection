"""
Merge Kaggle pallet dataset (already in YOLO format) into pallet_v1 structure.

This script:
1. Copies images from Kaggle train/valid/test to pallet_v1
2. Copies labels (ensuring class 0 = pallet)
3. Handles proper train/val/test split
"""

import shutil
from pathlib import Path


def verify_label_format(label_path):
    """Verify label file uses class 0 and has valid YOLO format."""
    try:
        with open(label_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    return False, f"Invalid line format: {line}"
                cls, xc, yc, w, h = parts
                # Ensure class is 0 (pallet)
                if cls != "0":
                    return False, f"Class is {cls}, expected 0"
                # Validate coordinates
                try:
                    coords = [float(xc), float(yc), float(w), float(h)]
                    if any(v < 0 or v > 1 for v in coords):
                        return False, f"Coordinates out of range: {coords}"
                except ValueError:
                    return False, f"Invalid coordinate values: {line}"
        return True, "OK"
    except Exception as e:
        return False, f"Error reading file: {e}"


def copy_kaggle_split(kaggle_split_dir, out_img_dir, out_label_dir, split_name):
    """
    Copy one split from Kaggle dataset to pallet_v1.
    
    Args:
        kaggle_split_dir: Path to Kaggle split (e.g., dataset/kaggleyolo8_palett/train)
        out_img_dir: Output images directory
        out_label_dir: Output labels directory
        split_name: Name for logging
    """
    kaggle_split_dir = Path(kaggle_split_dir)
    out_img_dir = Path(out_img_dir)
    out_label_dir = Path(out_label_dir)
    
    if not kaggle_split_dir.exists():
        print(f"  WARNING: Kaggle {split_name} directory not found: {kaggle_split_dir}")
        return 0
    
    # Find images and labels
    img_dir = kaggle_split_dir / "images"
    label_dir = kaggle_split_dir / "labels"
    
    if not img_dir.exists():
        print(f"  WARNING: Images directory not found: {img_dir}")
        return 0
    
    if not label_dir.exists():
        print(f"  WARNING: Labels directory not found: {label_dir}")
        return 0
    
    # Create output directories
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    image_files = [f for f in img_dir.iterdir() if f.suffix in image_extensions]
    
    copied_count = 0
    error_count = 0
    
    for img_file in image_files:
        # Find corresponding label file
        label_file = label_dir / (img_file.stem + ".txt")
        
        if not label_file.exists():
            print(f"  WARNING: Label not found for {img_file.name}, skipping")
            error_count += 1
            continue
        
        # Verify label format (class 0)
        is_valid, msg = verify_label_format(label_file)
        if not is_valid:
            print(f"  WARNING: Invalid label {label_file.name}: {msg}")
            # Still copy but note the issue
            error_count += 1
        
        # Copy image
        dst_img = out_img_dir / img_file.name
        shutil.copy2(img_file, dst_img)
        
        # Copy label
        dst_label = out_label_dir / label_file.name
        shutil.copy2(label_file, dst_label)
        
        copied_count += 1
    
    print(f"  Copied: {copied_count} images")
    if error_count > 0:
        print(f"  Warnings: {error_count} files had issues")
    
    return copied_count


def main():
    """Main merge function."""
    kaggle_root = Path("dataset/kaggleyolo8_palett")
    out_root = Path("datasets/pallet_v1")
    
    if not kaggle_root.exists():
        print(f"ERROR: Kaggle dataset not found at {kaggle_root}")
        print("\nPlease ensure Kaggle dataset is at: dataset/kaggleyolo8_palett/")
        return
    
    print("=== Merging Kaggle Dataset ===\n")
    
    # Map Kaggle splits to output splits
    # Kaggle uses: train, valid, test
    # We use: train, val, test
    splits = [
        ("train", "train", "train"),
        ("valid", "val", "val"),  # Kaggle "valid" -> our "val"
        ("test", "test", "test"),
    ]
    
    total_copied = 0
    for kaggle_split, out_split, display_name in splits:
        print(f"Processing {display_name} split...")
        kaggle_dir = kaggle_root / kaggle_split
        out_img_dir = out_root / "images" / out_split
        out_label_dir = out_root / "labels" / out_split
        
        count = copy_kaggle_split(kaggle_dir, out_img_dir, out_label_dir, display_name)
        total_copied += count
    
    print(f"\n=== Merge Complete ===")
    print(f"Total images copied: {total_copied}")


if __name__ == "__main__":
    main()
