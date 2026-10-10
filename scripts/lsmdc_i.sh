#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DATA_PATH="${DATA_PATH:-/mnt/data/Panzhenyi/train/data/LSMDC/LSMDC}"
export VIDEO_PATH="${VIDEO_PATH:-${DATA_PATH}/LSMDC_Videos}"
export VIDEO_PATH_JSON="${VIDEO_PATH_JSON:-${DATA_PATH}/video_path_LSMDC_Videos.json}"
export REBUILD_VIDEO_PATH_JSON="${REBUILD_VIDEO_PATH_JSON:-auto}"
export OUTPUT_PATH="${OUTPUT_PATH:-${SCRIPT_DIR}/../log/LSMDC_Videos_infer}"
export CKPT_PATH="${CKPT_PATH:-${SCRIPT_DIR}/../log/LSMDC_Videos/sap/best.pth}"
export PRETRAINED_PATH="${PRETRAINED_PATH:-/mnt/data/Panzhenyi/train/model}"
export BASE_ENCODER="${BASE_ENCODER:-ViT-B/32}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export BATCH_SIZE="${BATCH_SIZE:-256}"
export BATCH_SIZE_VAL="${BATCH_SIZE_VAL:-256}"

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
