#!/usr/bin/env bash
# Installs the third-party packages used by the data preprocessing pipeline into the active
# conda environment (created from environment_preprocess.yml).
#
#   conda env create -f environment_preprocess.yml
#   conda activate egoavflow-preprocess
#   bash scripts/install_preprocessing.sh
#
# CUDA extensions are compiled for TORCH_CUDA_ARCH_LIST, so no GPU is needed during installation.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "Activate the egoavflow-preprocess conda environment first." >&2
    exit 1
fi

export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5;8.0;8.6;8.9;9.0}"
export FORCE_CUDA=1
export MAX_JOBS="${MAX_JOBS:-8}"
# GroundingDINO only builds its CUDA op when this is set
export BUILD_WITH_CUDA=True

# Keep the versions from environment_preprocess.yml when building the packages below.
CONSTRAINTS="$(mktemp)"
trap 'rm -f "$CONSTRAINTS"' EXIT
python - "$CONSTRAINTS" <<'EOF'
import sys
from importlib.metadata import version
pins = ["torch", "torchvision", "numpy", "opencv-python", "transformers", "huggingface-hub", "diffusers"]
open(sys.argv[1], "w").write("".join(f"{p}=={version(p)}\n" for p in pins))
EOF
export PIP_CONSTRAINT="$CONSTRAINTS"

echo "==> Submodules"
git submodule update --init third_party/Cutie third_party/Grounded-Segment-Anything
git submodule update --init --recursive third_party/hamer third_party/DROID-SLAM

echo "==> egoavflow and CoTracker3"
pip install -e . -e third_party/cotracker3

echo "==> HaMeR (hand pose)"
pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2@a9c0821a12ad353fb2a96f019515990d5460c5ac"
pip install --no-build-isolation "git+https://github.com/mattloper/chumpy@580566eafc9ac68b2614b64d6f7aaa84eebb70da"
pip install --no-build-isolation mmcv==1.3.9
pip install --no-deps -e third_party/hamer
pip install --no-deps --no-build-isolation -e third_party/hamer/third-party/ViTPose

echo "==> GroundingDINO, Segment Anything, Cutie (object and hand masks)"
pip install --no-deps -e third_party/Grounded-Segment-Anything/segment_anything
pip install --no-deps --no-build-isolation -e third_party/Grounded-Segment-Anything/GroundingDINO
pip install --no-deps -e third_party/Cutie

echo "==> DROID-SLAM (camera trajectory)"
pip install --no-build-isolation third_party/DROID-SLAM/thirdparty/lietorch
# prebuilt torch-scatter for torch 2.5.1 + CUDA 12.1 (building third_party/DROID-SLAM/thirdparty/pytorch_scatter also works, but slowly)
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip install --no-deps --no-build-isolation -e third_party/DROID-SLAM

echo "==> Checking imports"
python - <<'EOF'
import torch, detectron2, hamer, mmpose, groundingdino, segment_anything, cutie
import lietorch, droid_backends, torch_scatter, gtsam, diffusers, xformers, cotracker
import hamer.datasets.vitdet_dataset, hamer.utils.renderer
from groundingdino import _C  # CUDA op of GroundingDINO
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| diffusers", diffusers.__version__)
EOF
echo "Done."
