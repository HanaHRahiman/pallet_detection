from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_OUTPUT = Path("II/OBB/dist/obb_colab_bundle")
DEFAULT_DATASET = Path("II/OBB/data/roboflo2_pallet_obb_v3")


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing required path: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing required file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Package the minimal OBB training bundle for Google Colab.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Dataset root to bundle.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output directory for the bundle.")
    parser.add_argument("--zip-name", default="obb_colab_bundle.zip", help="Zip filename to create inside the parent directory.")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    output = Path(args.output)
    zip_name = args.zip_name

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    # Keep the bundle minimal: no reports, no converted-only review folder.
    copy_tree(dataset / "train", output / "II/OBB/data/roboflo2_pallet_obb_v3/train")
    copy_tree(dataset / "valid", output / "II/OBB/data/roboflo2_pallet_obb_v3/valid")
    copy_tree(dataset / "test", output / "II/OBB/data/roboflo2_pallet_obb_v3/test")
    copy_file(dataset / "data.yaml", output / "II/OBB/data/roboflo2_pallet_obb_v3/data.yaml")

    copy_file(Path("II/OBB/configs/train_obb_yolo11m.yaml"), output / "II/OBB/configs/train_obb_yolo11m.yaml")
    copy_file(Path("II/OBB/scripts/train_pallet_obb.py"), output / "II/OBB/scripts/train_pallet_obb.py")
    copy_file(Path("II/requirements-colab.txt"), output / "II/requirements-colab.txt")

    archive_base = output.parent / Path(zip_name).stem
    archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=output)

    size_mb = Path(archive_path).stat().st_size / (1024 * 1024)
    print(f"[DONE] Bundle folder: {output.resolve()}")
    print(f"[DONE] Bundle zip: {Path(archive_path).resolve()} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
