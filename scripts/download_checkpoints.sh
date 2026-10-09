#!/usr/bin/env bash
# Downloads the pretrained third-party weights.
#
#   bash scripts/download_checkpoints.sh eval        # offline evaluation only (CoTracker3)
#   bash scripts/download_checkpoints.sh preprocess  # evaluation + data preprocessing
#
# Weights go to $EGOAVFLOW_CHECKPOINT_ROOT (default: <repo>/checkpoints), except for HaMeR and Cutie,
# which are placed where those packages look for them (third_party/hamer/_DATA, third_party/Cutie/weights).
# The preprocessing weights need the egoavflow-preprocess environment (gdown, huggingface-cli, cutie).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="${EGOAVFLOW_CHECKPOINT_ROOT:-$ROOT/checkpoints}"
TARGET="${1:-eval}"

download() {
    local url="$1" out="$2"
    if [[ -f "$out" ]]; then
        echo "exists: $out"
        return
    fi
    mkdir -p "$(dirname "$out")"
    wget -q --show-progress -O "$out.part" "$url"
    mv "$out.part" "$out"
}

echo "==> CoTracker3 (online)"
download https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth "$CKPT/cotracker3/scaled_online.pth"

if [[ "$TARGET" == "eval" ]]; then
    exit 0
elif [[ "$TARGET" != "preprocess" ]]; then
    echo "usage: $0 [eval|preprocess]" >&2
    exit 1
fi

echo "==> GroundingDINO (Swin-T) and Segment Anything (ViT-H)"
download https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
    "$CKPT/grounding_dino/groundingdino_swint_ogc.pth"
download https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth "$CKPT/sam/sam_vit_h_4b8939.pth"

echo "==> DROID-SLAM"
if [[ ! -f "$CKPT/droid/droid.pth" ]]; then
    mkdir -p "$CKPT/droid"
    gdown 1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh -O "$CKPT/droid/droid.pth"
fi

echo "==> Stable Diffusion 2.1 (community mirror of the deprecated stabilityai/stable-diffusion-2-1)"
huggingface-cli download sd2-community/stable-diffusion-2-1 \
    --revision bb2154823665391b4fb29b0b9cf82a198964ee05 \
    --local-dir "$CKPT/stable-diffusion-2-1" \
    --include "model_index.json" "scheduler/*" "tokenizer/*" "feature_extractor/*" \
    "text_encoder/config.json" "text_encoder/model.safetensors" \
    "unet/config.json" "unet/diffusion_pytorch_model.safetensors" \
    "vae/config.json" "vae/diffusion_pytorch_model.safetensors"

echo "==> HaMeR and ViTPose"
HAMER="$ROOT/third_party/hamer"
if [[ ! -f "$HAMER/_DATA/hamer_ckpts/checkpoints/hamer.ckpt" ]]; then
    wget -q --show-progress -O "$HAMER/hamer_demo_data.tar.gz" https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz
    tar --warning=no-unknown-keyword --exclude=".*" -xf "$HAMER/hamer_demo_data.tar.gz" -C "$HAMER"
    rm "$HAMER/hamer_demo_data.tar.gz"
fi

echo "==> Cutie"
python -c "from cutie.utils.download_models import download_models_if_needed; download_models_if_needed()"

if [[ ! -f "$HAMER/_DATA/data/mano/MANO_RIGHT.pkl" ]]; then
    echo
    echo "MANO is not redistributable: register at https://mano.is.tue.mpg.de, download the MANO model,"
    echo "and copy MANO_RIGHT.pkl to $HAMER/_DATA/data/mano/MANO_RIGHT.pkl"
fi
echo "Done. (The ViTDet detector weights and GroundingDINO's BERT tokenizer are downloaded on first use.)"
