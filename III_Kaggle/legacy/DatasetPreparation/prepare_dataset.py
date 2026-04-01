"""
Main dataset preparation script.

Orchestrates the entire dataset preparation process:
1. Convert LOCO COCO format to pallet-only YOLO
2. Merge Kaggle YOLO dataset
3. Run sanity checks
4. Create data.yaml

Usage:
    python datasets/prepare_dataset.py
"""

import sys
from pathlib import Path

# Add parent directory to path to import our modules
sys.path.insert(0, str(Path(__file__).parent))

from convert_loco_to_yolo import main as convert_loco
from merge_kaggle_dataset import main as merge_kaggle
from sanity_check import main as sanity_check


def create_data_yaml():
    """Create data.yaml configuration file."""
    yaml_content = """path: datasets/pallet_v1
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: pallet
"""
    
    # Use relative path from project root
    yaml_path = Path(__file__).parent.parent.parent.parent / "datasets" / "pallet_v1" / "data.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    
    print(f"✓ Created data.yaml at {yaml_path}")


def main():
    """Main orchestration function."""
    print("=" * 60)
    print("PALLET DATASET PREPARATION")
    print("=" * 60)
    print("\nThis script will:")
    print("  1. Convert LOCO COCO format → pallet-only YOLO")
    print("  2. Merge Kaggle YOLO dataset")
    print("  3. Run sanity checks")
    print("  4. Create data.yaml")
    print()
    
    # Step 1: Convert LOCO
    print("\n" + "=" * 60)
    print("STEP 1: Converting LOCO dataset")
    print("=" * 60)
    try:
        convert_loco()
    except Exception as e:
        print(f"\nERROR in LOCO conversion: {e}")
        print("\nIf LOCO dataset is not available yet, you can:")
        print("  - Skip this step and continue with Kaggle only")
        print("  - Download LOCO and place it in: datasets/raw/LOCO/")
        response = input("\nContinue with Kaggle dataset only? (y/n): ")
        if response.lower() != 'y':
            print("Exiting. Please set up LOCO dataset first.")
            return
    
    # Step 2: Merge Kaggle
    print("\n" + "=" * 60)
    print("STEP 2: Merging Kaggle dataset")
    print("=" * 60)
    try:
        merge_kaggle()
    except Exception as e:
        print(f"\nERROR in Kaggle merge: {e}")
        return
    
    # Step 3: Sanity check
    print("\n" + "=" * 60)
    print("STEP 3: Running sanity checks")
    print("=" * 60)
    try:
        sanity_check()
    except Exception as e:
        print(f"\nERROR in sanity check: {e}")
        return
    
    # Step 4: Create data.yaml
    print("\n" + "=" * 60)
    print("STEP 4: Creating data.yaml")
    print("=" * 60)
    create_data_yaml()
    
    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Review sanity check results above")
    print("  2. If all checks passed, train with:")
    print("     yolo detect train model=yolov8s.pt data=datasets/pallet_v1/data.yaml imgsz=960 epochs=80 batch=16")
    print()


if __name__ == "__main__":
    main()
