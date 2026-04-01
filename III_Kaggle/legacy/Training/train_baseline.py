"""
YOLOv8 Baseline Training Script

Trains YOLOv8s on the pallet_v1 dataset.
This is the baseline model before adding CCTV and NVIDIA synthetic data.

Usage:
    python datasets/train_baseline.py
"""

from ultralytics import YOLO
from pathlib import Path
import torch


def main():
    """Train YOLOv8 baseline model."""
    # Model configuration
    model_name = "yolov8s.pt"  # Start with small model, can upgrade to yolov8m.pt later
    # Path relative to project root
    data_yaml = str(Path(__file__).parent.parent.parent.parent / "datasets" / "pallet_v1" / "data.yaml")
    img_size = 960  # Larger size helps detect small pallets in distance
    epochs = 80
    batch_size = 16
    
    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
    else:
        print("⚠ No GPU detected - using CPU (training will be slower)")
        print("  Consider reducing batch_size or img_size for faster CPU training")
    
    # Verify data.yaml exists
    if not Path(data_yaml).exists():
        print(f"ERROR: data.yaml not found at {data_yaml}")
        print("Please run datasets/prepare_dataset.py first")
        return
    
    print("=" * 60)
    print("YOLOv8 BASELINE TRAINING")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Data: {data_yaml}")
    print(f"Device: {device.upper()}")
    print(f"Image size: {img_size}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print()
    
    # Load model
    model = YOLO(model_name)
    
    # Train with explicit device specification
    try:
        results = model.train(
            data=data_yaml,
            imgsz=img_size,
            epochs=epochs,
            batch=batch_size,
            device=device,  # Explicitly set device
            name="pallet_baseline",
            project="runs/detect",
            save=True,
            plots=True,
            verbose=True,  # Show detailed progress
        )
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        print("\nTroubleshooting:")
        print("1. Check if dataset is accessible")
        print("2. Try reducing batch_size (e.g., batch_size=8)")
        print("3. Check available disk space")
        raise
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nBest model saved at: runs/detect/pallet_baseline/weights/best.pt")
    print(f"\nTo evaluate:")
    print(f"  yolo detect val model=runs/detect/pallet_baseline/weights/best.pt data={data_yaml}")
    print()


if __name__ == "__main__":
    main()
