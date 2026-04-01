"""
Google Colab Training Script - Can be run locally or in Colab

This script does the same thing as train_colab.ipynb but as a Python script.
You can edit this in Cursor, then copy the code to Colab cells.

Usage in Colab:
1. Copy sections into Colab cells
2. Or upload this file and run: exec(open('train_colab_script.py').read())
"""

import torch
from pathlib import Path

# ============================================================================
# STEP 1: Check GPU (Run this first in Colab)
# ============================================================================
def check_gpu():
    """Check if GPU is available."""
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    return torch.cuda.is_available()

# ============================================================================
# STEP 2: Install Dependencies (Run in Colab)
# ============================================================================
# In Colab, run: !pip install ultralytics -q

# ============================================================================
# STEP 3: Setup Dataset (Run in Colab)
# ============================================================================
def setup_dataset_from_drive(drive_path="/content/drive/MyDrive/content/pallet_v1", 
                              local_path="/content/pallet_v1"):
    """
    Copy dataset from Google Drive to local Colab storage.
    
    Run this in Colab:
    from google.colab import drive
    drive.mount('/content/drive')
    setup_dataset_from_drive()
    """
    import shutil
    
    if Path(local_path).exists():
        print(f"✓ Dataset already exists at {local_path}")
        return True
    
    if Path(drive_path).exists():
        print(f"Copying dataset from {drive_path} to {local_path}...")
        shutil.copytree(drive_path, local_path)
        print("✓ Dataset copied!")
        return True
    else:
        print(f"❌ Dataset not found at {drive_path}")
        print("Please check the path or upload dataset directly to Colab")
        return False

# ============================================================================
# STEP 4: Fix data.yaml Path (Run in Colab)
# ============================================================================
def fix_data_yaml(data_yaml_path="/content/pallet_v1/data.yaml"):
    """Fix data.yaml paths for Colab."""
    import yaml
    from pathlib import Path
    
    if not Path(data_yaml_path).exists():
        print(f"❌ data.yaml not found at {data_yaml_path}")
        return False
    
    # Read current data.yaml
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"Current path in data.yaml: {data_config.get('path', 'NOT SET')}")
    
    # Update path to Colab location
    data_config['path'] = '/content/pallet_v1'
    
    # Write back
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)
    
    print("✓ Updated data.yaml for Colab paths")
    print(f"  New path: {data_config['path']}")
    
    # Verify images exist
    train_path = Path(data_config['path']) / data_config.get('train', 'images/train')
    val_path = Path(data_config['path']) / data_config.get('val', 'images/val')
    
    if train_path.exists():
        img_count = len(list(train_path.glob('*.jpg'))) + len(list(train_path.glob('*.png')))
        print(f"  ✓ Train images found: {img_count} files")
    else:
        print(f"  ⚠ Train images NOT found at: {train_path}")
    
    if val_path.exists():
        img_count = len(list(val_path.glob('*.jpg'))) + len(list(val_path.glob('*.png')))
        print(f"  ✓ Val images found: {img_count} files")
    else:
        print(f"  ⚠ Val images NOT found at: {val_path}")
    
    return train_path.exists() and val_path.exists()

# ============================================================================
# STEP 5: Train Model (Run in Colab)
# ============================================================================
def train_model(data_yaml="/content/pallet_v1/data.yaml",
                model_name="yolov8s.pt",
                img_size=960,
                epochs=80,
                batch_size=16):
    """Train YOLOv8 model."""
    from ultralytics import YOLO
    from pathlib import Path
    import yaml
    
    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Verify dataset
    print(f"\n=== Dataset Verification ===")
    if not Path(data_yaml).exists():
        print(f"❌ ERROR: data.yaml not found at {data_yaml}")
        return None
    
    print(f"✓ data.yaml found")
    
    # Double-check paths
    with open(data_yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    base_path = Path(config['path'])
    train_path = base_path / config.get('train', 'images/train')
    val_path = base_path / config.get('val', 'images/val')
    
    print(f"  Base path: {base_path}")
    print(f"  Train path: {train_path} - {'✓ EXISTS' if train_path.exists() else '❌ MISSING'}")
    print(f"  Val path: {val_path} - {'✓ EXISTS' if val_path.exists() else '❌ MISSING'}")
    
    if not (train_path.exists() and val_path.exists()):
        print(f"\n❌ Dataset structure incorrect!")
        return None
    
    print(f"\n✓ Starting training...")
    print(f"Model: {model_name}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {img_size}")
    
    model = YOLO(model_name)
    results = model.train(
        data=data_yaml,
        imgsz=img_size,
        epochs=epochs,
        batch=batch_size,
        device=device,
        name="pallet_baseline_colab",
        project="/content/runs/detect",
        save=True,
        plots=True,
        verbose=True,
    )
    
    print("\n✓ Training complete!")
    print(f"Model: /content/runs/detect/pallet_baseline_colab/weights/best.pt")
    
    return results

# ============================================================================
# STEP 6: Download Model (Run in Colab)
# ============================================================================
def download_model(model_path="/content/runs/detect/pallet_baseline_colab/weights/best.pt"):
    """Download trained model from Colab."""
    from google.colab import files
    from pathlib import Path
    
    if Path(model_path).exists():
        files.download(model_path)
        print("✓ Model downloaded!")
        return True
    else:
        print("❌ Model not found")
        return False

# ============================================================================
# MAIN EXECUTION (For Colab)
# ============================================================================
if __name__ == "__main__":
    # This will run if you execute the script in Colab
    # Uncomment the steps you want to run:
    
    # Step 1: Check GPU
    check_gpu()
    
    # Step 2: Install (run separately): !pip install ultralytics -q
    
    # Step 3: Setup dataset (after mounting Drive)
    # setup_dataset_from_drive()
    
    # Step 4: Fix data.yaml
    # fix_data_yaml()
    
    # Step 5: Train
    # train_model()
    
    # Step 6: Download
    # download_model()
    
    print("\n" + "="*60)
    print("To use in Colab:")
    print("1. Upload this file to Colab")
    print("2. Run: exec(open('train_colab_script.py').read())")
    print("3. Or copy individual functions into Colab cells")
    print("="*60)
