#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DATA_PATH="${DATA_PATH:-/mnt/data/Panzhenyi/train/data/LSMDC/LSMDC}"
export VIDEO_PATH="${VIDEO_PATH:-/mnt/data/Panzhenyi/train/data/LSMDC/LSMDC/LSMDC_Videos_mp4}"
export PRETRAINED_PATH="${PRETRAINED_PATH:-/mnt/data/Panzhenyi/train/model}"
export BASE_ENCODER="${BASE_ENCODER:-ViT-B/16}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export CKPT_PATH="${CKPT_PATH:-${SCRIPT_DIR}/../log/LSMDC/best.pth}"

export USE_CLAA="${USE_CLAA:-1}"
export USE_SAP="${USE_SAP:-1}"
export USE_PAMR="${USE_PAMR:-0}"
export USE_CLIP_REFINER="${USE_CLIP_REFINER:-0}"

export RESCLIP_WINDOW="${RESCLIP_WINDOW:-4}"
export CLAA_TAU="${CLAA_TAU:-2.0}"
export CLAA_LAMBDA="${CLAA_LAMBDA:-0.6}"
export SAP_TAU="${SAP_TAU:-2.1}"
export SAP_LAMBDA="${SAP_LAMBDA:-0.6}"
export SAP_TEMP_THD="${SAP_TEMP_THD:-0.0}"

exec "${SCRIPT_DIR}/LSMDC_infer.sh" "$@"
