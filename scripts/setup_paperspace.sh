#!/usr/bin/env bash
# Paperspace Gradient / Notebook machine — first-time setup
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Project: $ROOT"
echo "==> Python: $(python3 --version 2>&1 || true)"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

echo "==> Verifying TensorFlow GPU (optional)..."
python -c "
import tensorflow as tf
print('TensorFlow', tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print('GPUs:', gpus if gpus else 'none (CPU only)')
"

echo "==> Verifying model shapes..."
python scripts/verify_shapes.py

echo ""
echo "Done. Next steps:"
echo "  1. Upload GDF files to: data/raw/BCICIV_2a_gdf/"
echo "  2. source .venv/bin/activate"
echo "  3. python scripts/run_subject_dependent.py"
