#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash III_kaggle/scripts/run_colab_yolov8m_80e.sh <data_yaml_path> [project_dir]"
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_SOURCE="$1"
PROJECT_DIR="${2:-/content/runs/detect}"

cd "$REPO_ROOT"

python III_kaggle/scripts/train_kaggle_baseline.py \
  --config III_kaggle/configs/train_detect_kaggle_yolov8m_80e.yaml \
  --data-source "$DATA_SOURCE" \
  --project "$PROJECT_DIR" \
  --run-meta-out III_kaggle/reports/kaggle_baseline/train_run_yolov8m_80e.json
