#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/data/PZY/train/TempMe-Rf}"
DATA_PATH="${DATA_PATH:-/mnt/data/PZY/train/CLIP4Clip-master/data/msvd/msvd_data}"
VIDEO_PATH="${VIDEO_PATH:-${DATA_PATH}/YouTubeClips/YouTubeClips}"
CKPT_PATH="${CKPT_PATH:-${PROJECT_DIR}/log/MSVD/best.pth}"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_DIR}/log/MSVD_infer}"
PRETRAINED_PATH="${PRETRAINED_PATH:-/mnt/data/PZY/train/DGL-main/DGL-main/modules/pretrained}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/data/PZY/anaconda3/envs/DL/bin/python}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,4,5}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29505}"

WORKERS="${WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-128}"
BATCH_SIZE_VAL="${BATCH_SIZE_VAL:-128}"
MAX_WORDS="${MAX_WORDS:-32}"
MAX_FRAMES="${MAX_FRAMES:-12}"
VIDEO_FRAMERATE="${VIDEO_FRAMERATE:-1}"
CLIP_LR="${CLIP_LR:-6e-5}"
N_DISPLAY="${N_DISPLAY:-50}"

BASE_ENCODER="${BASE_ENCODER:-ViT-B/32}"
LORA_DIM="${LORA_DIM:-8}"
TOME_R="${TOME_R:-2}"
MERGE_LAYER="${MERGE_LAYER:-8-9-10}"
MERGE_FRAME_NUM="${MERGE_FRAME_NUM:-2-2-3}"
MERGE_TOKEN_PROPORTION="${MERGE_TOKEN_PROPORTION:-30-10}"
FRAME_POS="${FRAME_POS:-1}"

USE_CLAA="${USE_CLAA:-1}"
USE_SAP="${USE_SAP:-1}"
USE_PAMR="${USE_PAMR:-0}"
RESCLIP_WINDOW="${RESCLIP_WINDOW:-2}"
CLAA_TAU="${CLAA_TAU:-1.0}"
CLAA_LAMBDA="${CLAA_LAMBDA:-0.15}"
SAP_TAU="${SAP_TAU:-1.0}"
SAP_LAMBDA="${SAP_LAMBDA:-0.10}"
SAP_TEMP_THD="${SAP_TEMP_THD:-0.0}"
PAMR_STEPS="${PAMR_STEPS:-10}"
PAMR_STRIDE="${PAMR_STRIDE:-8-16}"
USE_CLIP_REFINER="${USE_CLIP_REFINER:-0}"
CLIP_REFINER_DEPTH="${CLIP_REFINER_DEPTH:-2}"
CLIP_REFINER_HIDDEN_RATIO="${CLIP_REFINER_HIDDEN_RATIO:-0.25}"
CLIP_REFINER_RESIDUAL_SCALE="${CLIP_REFINER_RESIDUAL_SCALE:-0.10}"
CLIP_REFINER_TEMPERATURE="${CLIP_REFINER_TEMPERATURE:-0.07}"
CLIP_REFINER_LAMBDA="${CLIP_REFINER_LAMBDA:-0.05}"
CLIP_REFINER_SCD_WEIGHT="${CLIP_REFINER_SCD_WEIGHT:-0.005}"

required_files=(
  "${DATA_PATH}/train_list.txt"
  "${DATA_PATH}/val_list.txt"
  "${DATA_PATH}/test_list.txt"
  "${DATA_PATH}/raw-captions.pkl"
  "${CKPT_PATH}"
  "${PRETRAINED_PATH}/ViT-B-32.pt"
)

for path in "${required_files[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
done

if [[ ! -d "${VIDEO_PATH}" ]]; then
  echo "Missing MSVD video directory: ${VIDEO_PATH}" >&2
  exit 1
fi

"${PYTHON_BIN}" - <<'PY'
missing = []
for module in ("decord", "torchvision"):
    try:
        __import__(module)
    except Exception:
        missing.append(module)
if missing:
    raise SystemExit(
        "Missing Python package(s): {}. Install them in the active environment before inference.".format(
            ", ".join(missing)
        )
    )
PY

MAX_FRAMES="${MAX_FRAMES}" MERGE_FRAME_NUM="${MERGE_FRAME_NUM}" "${PYTHON_BIN}" - <<'PY'
import os

frame_num = int(os.environ["MAX_FRAMES"])
merge_frame_num = [int(item) for item in os.environ["MERGE_FRAME_NUM"].split("-")]
original_frame_num = frame_num
for item in merge_frame_num:
    if item <= 0:
        raise SystemExit("MERGE_FRAME_NUM must contain positive integers.")
    if frame_num % item != 0:
        raise SystemExit(
            "Invalid MERGE_FRAME_NUM={} for MAX_FRAMES={}: {} cannot be divided by {}. "
            "Use 2-2-3 for 12-frame MSVD A0.".format(
                os.environ["MERGE_FRAME_NUM"], original_frame_num, frame_num, item
            )
        )
    frame_num //= item

if frame_num != 1:
    raise SystemExit(
        "A0 holistic merging should end at 1 clip, but MAX_FRAMES={} and MERGE_FRAME_NUM={} end at {}.".format(
            original_frame_num, os.environ["MERGE_FRAME_NUM"], frame_num
        )
    )
PY

mkdir -p "${OUTPUT_PATH}"
cd "${PROJECT_DIR}"

echo "Inferring TempMe-RES-refine on MSVD test split"
echo "  project: ${PROJECT_DIR}"
echo "  data: ${DATA_PATH}"
echo "  videos: ${VIDEO_PATH}"
echo "  checkpoint: ${CKPT_PATH}"
echo "  output: ${OUTPUT_PATH}"
echo "  eval split: ${EVAL_SPLIT}"
echo "  pretrained: ${PRETRAINED_PATH}/ViT-B-32.pt"
echo "  used gpus: ${CUDA_VISIBLE_DEVICES}"
echo "  gpus/processes: ${NPROC_PER_NODE}"
echo "  A0 merge layer: ${MERGE_LAYER}"
echo "  A0 merge frame num: ${MERGE_FRAME_NUM}"
echo "  A0 token removal: ${MERGE_TOKEN_PROPORTION}"

"${PYTHON_BIN}" -m torch.distributed.launch \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  main.py \
  --do_eval 1 \
  --workers "${WORKERS}" \
  --anno_path "${DATA_PATH}" \
  --video_path "${VIDEO_PATH}" \
  --datatype msvd \
  --output_dir "${OUTPUT_PATH}" \
  --pretrained_path "${PRETRAINED_PATH}" \
  --eval_split "${EVAL_SPLIT}" \
  --init_model "${CKPT_PATH}" \
  --batch_size "${BATCH_SIZE}" \
  --batch_size_val "${BATCH_SIZE_VAL}" \
  --max_words "${MAX_WORDS}" \
  --max_frames "${MAX_FRAMES}" \
  --video_framerate "${VIDEO_FRAMERATE}" \
  --clip_lr "${CLIP_LR}" \
  --n_display "${N_DISPLAY}" \
  --base_encoder "${BASE_ENCODER}" \
  --lora_dim "${LORA_DIM}" \
  --tome_r "${TOME_R}" \
  --merge_layer "${MERGE_LAYER}" \
  --merge_frame_num "${MERGE_FRAME_NUM}" \
  --merge_token_proportion "${MERGE_TOKEN_PROPORTION}" \
  --frame_pos "${FRAME_POS}" \
  --use_claa "${USE_CLAA}" \
  --use_sap "${USE_SAP}" \
  --use_pamr "${USE_PAMR}" \
  --resclip_window "${RESCLIP_WINDOW}" \
  --claa_tau "${CLAA_TAU}" \
  --claa_lambda "${CLAA_LAMBDA}" \
  --sap_tau "${SAP_TAU}" \
  --sap_lambda "${SAP_LAMBDA}" \
  --sap_temp_thd "${SAP_TEMP_THD}" \
  --pamr_steps "${PAMR_STEPS}" \
  --pamr_stride "${PAMR_STRIDE}" \
  --use_clip_refiner "${USE_CLIP_REFINER}" \
  --clip_refiner_depth "${CLIP_REFINER_DEPTH}" \
  --clip_refiner_hidden_ratio "${CLIP_REFINER_HIDDEN_RATIO}" \
  --clip_refiner_residual_scale "${CLIP_REFINER_RESIDUAL_SCALE}" \
  --clip_refiner_temperature "${CLIP_REFINER_TEMPERATURE}" \
  --clip_refiner_lambda "${CLIP_REFINER_LAMBDA}" \
  --clip_refiner_scd_weight "${CLIP_REFINER_SCD_WEIGHT}" \
  "$@"
