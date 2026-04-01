"""
Dataset sanity check script.

Validates:
1. Every image has a corresponding label file
2. YOLO boxes are within [0,1]
3. Label format is correct (5 values per line)
4. No empty/corrupt images
"""

from pathlib import Path
from PIL import Image


def check_split(root, split_name):
    """Check one split (train/val/test)."""
    img_dir = root / "images" / split_name
    label_dir = root / "labels" / split_name
    
    if not img_dir.exists():
        print(f"  ERROR: Images directory not found: {img_dir}")
        return 0, 0
    
    if not label_dir.exists():
        print(f"  ERROR: Labels directory not found: {label_dir}")
        return 0, 0
    
    # Get all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    image_files = list(img_dir.glob("*.*"))
    image_files = [f for f in image_files if f.suffix in image_extensions]
    
    issues = []
    checked_count = 0
    
    for img_file in image_files:
        checked_count += 1
        label_file = label_dir / (img_file.stem + ".txt")
        
        # Check 1: Label file exists
        if not label_file.exists():
            issues.append(f"MISSING LABEL: {img_file.name}")
            continue
        
        # Check 2: Image is valid
        try:
            img = Image.open(img_file)
            img.verify()  # Verify it's a valid image
        except Exception as e:
            issues.append(f"CORRUPT IMAGE: {img_file.name} - {e}")
            continue
        
        # Check 3: Label format
        try:
            with open(label_file, "r") as f:
                lines = f.read().strip().splitlines()
                
                if not lines:
                    issues.append(f"EMPTY LABEL: {label_file.name}")
                    continue
                
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        issues.append(f"BAD LINE ({label_file.name}:{line_num}): {line}")
                        continue
                    
                    cls, xc, yc, w, h = parts
                    
                    # Check class is valid
                    try:
                        cls_int = int(cls)
                        if cls_int != 0:
                            issues.append(f"WRONG CLASS ({label_file.name}:{line_num}): class={cls_int}, expected 0")
                    except ValueError:
                        issues.append(f"INVALID CLASS ({label_file.name}:{line_num}): {cls}")
                    
                    # Check coordinates are valid floats and in range
                    try:
                        coords = [float(xc), float(yc), float(w), float(h)]
                        for i, (name, val) in enumerate(zip(["xc", "yc", "w", "h"], coords)):
                            if val < 0 or val > 1:
                                issues.append(f"OUT OF RANGE ({label_file.name}:{line_num}): {name}={val:.6f}")
                    except ValueError:
                        issues.append(f"INVALID COORDS ({label_file.name}:{line_num}): {line}")
        
        except Exception as e:
            issues.append(f"ERROR READING LABEL ({label_file.name}): {e}")
    
    return checked_count, len(issues), issues


def main():
    """Main sanity check function."""
    root = Path("datasets/pallet_v1")
    
    if not root.exists():
        print(f"ERROR: Dataset directory not found: {root}")
        return
    
    print("=== Dataset Sanity Check ===\n")
    
    splits = ["train", "val", "test"]
    total_checked = 0
    total_issues = 0
    all_issues = []
    
    for split in splits:
        print(f"Checking {split} split...")
        checked, issue_count, issues = check_split(root, split)
        total_checked += checked
        total_issues += issue_count
        
        if issue_count == 0:
            print(f"  ✓ {checked} images checked, no issues")
        else:
            print(f"  ✗ {checked} images checked, {issue_count} issues found")
            all_issues.extend(issues)
    
    print(f"\n=== Summary ===")
    print(f"Total images checked: {total_checked}")
    print(f"Total issues: {total_issues}")
    
    if all_issues:
        print(f"\n=== Issues Found ===")
        for issue in all_issues[:50]:  # Show first 50 issues
            print(f"  {issue}")
        if len(all_issues) > 50:
            print(f"  ... and {len(all_issues) - 50} more issues")
    else:
        print("\n✓ All checks passed! Dataset is ready for training.")


if __name__ == "__main__":
    main()
