#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATA_PATH="${DATA_PATH:-/mnt/data/Panzhenyi/train/data/LSMDC/LSMDC}"
VIDEO_PATH="${VIDEO_PATH:-${DATA_PATH}/LSMDC_Videos_mp4}"
VIDEO_PATH_JSON="${VIDEO_PATH_JSON:-${DATA_PATH}/video_path.json}"
REBUILD_VIDEO_PATH_JSON="${REBUILD_VIDEO_PATH_JSON:-auto}"
LSMDC_MIN_VIDEO_BYTES="${LSMDC_MIN_VIDEO_BYTES:-10240}"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_DIR}/log/LSMDC}"
PRETRAINED_PATH="${PRETRAINED_PATH:-/mnt/data/Panzhenyi/train/model}"
INIT_MODEL="${INIT_MODEL:-}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/data/Panzhenyi/DL/bin/python3}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,3}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export VIDEO_DECODER="${VIDEO_DECODER:-cv2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

NPROC_PER_NODE="${NPROC_PER_NODE:-${NPROC:-2}}"
VISIBLE_DEVICE_COUNT="${CUDA_VISIBLE_DEVICES//,/ }"
VISIBLE_DEVICE_COUNT="$(wc -w <<< "${VISIBLE_DEVICE_COUNT}")"
if [[ "${NPROC_PER_NODE}" -gt "${VISIBLE_DEVICE_COUNT}" ]]; then
  echo "NPROC_PER_NODE=${NPROC_PER_NODE}, but CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} exposes only ${VISIBLE_DEVICE_COUNT} device(s)." >&2
  echo "Use CUDA_VISIBLE_DEVICES=1,3 NPROC_PER_NODE=2 for 2-GPU LSMDC training." >&2
  exit 1
fi
MASTER_PORT="${MASTER_PORT:-29508}"
if [[ -z "${DIST_BACKEND:-}" ]]; then
  if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
    DIST_BACKEND="nccl"
  else
    DIST_BACKEND="gloo"
  fi
fi
DDP_MODE="${DDP_MODE:-manual}"
USE_AMP="${USE_AMP:-1}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"

WORKERS="${WORKERS:-8}"
EPOCHS="${EPOCHS:-5}"
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-16}"
PER_GPU_BATCH_SIZE_VAL="${PER_GPU_BATCH_SIZE_VAL:-16}"
BATCH_SIZE="${BATCH_SIZE:-64}"
BATCH_SIZE_VAL="${BATCH_SIZE_VAL:-64}"
MAX_WORDS="${MAX_WORDS:-32}"
MAX_FRAMES="${MAX_FRAMES:-12}"
VIDEO_FRAMERATE="${VIDEO_FRAMERATE:-1}"
CLIP_LR="${CLIP_LR:-6e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.2}"
WARMUP_PROPORTION="${WARMUP_PROPORTION:-0.1}"
N_DISPLAY="${N_DISPLAY:-40}"
BASE_ENCODER="${BASE_ENCODER:-ViT-B/32}"
LORA_DIM="${LORA_DIM:-8}"
TOME_R="${TOME_R:-2}"
MERGE_LAYER="${MERGE_LAYER:-8-9-10}"
MERGE_FRAME_NUM="${MERGE_FRAME_NUM:-2-2-3}"
FRAME_POS="${FRAME_POS:-1}"

case "${BASE_ENCODER}" in
  "ViT-B/32")
    PRETRAINED_MODEL="${PRETRAINED_PATH}/ViT-B-32.pt"
    MERGE_TOKEN_PROPORTION="${MERGE_TOKEN_PROPORTION:-30-10}"
    ;;
  "ViT-B/16")
    PRETRAINED_MODEL="${PRETRAINED_PATH}/ViT-B-16.pt"
    MERGE_TOKEN_PROPORTION="${MERGE_TOKEN_PROPORTION:-40-20}"
    ;;
  *)
    echo "Unsupported BASE_ENCODER: ${BASE_ENCODER}" >&2
    exit 1
    ;;
esac

# Enable Rf refinements by default for LSMDC training.
USE_CLAA="${USE_CLAA:-0}"
USE_SAP="${USE_SAP:-0}"
USE_PAMR="${USE_PAMR:-0}"
RESCLIP_WINDOW="${RESCLIP_WINDOW:-8}"
CLAA_TAU="${CLAA_TAU:-2.0}"
CLAA_LAMBDA="${CLAA_LAMBDA:-0.6}"
SAP_TAU="${SAP_TAU:-2.1}"
SAP_LAMBDA="${SAP_LAMBDA:-0.4}"
SAP_TEMP_THD="${SAP_TEMP_THD:-0.0}"
PAMR_STEPS="${PAMR_STEPS:-10}"
PAMR_STRIDE="${PAMR_STRIDE:-8-16}"
USE_CLIP_REFINER="${USE_CLIP_REFINER:-0}"
CLIP_REFINER_DEPTH="${CLIP_REFINER_DEPTH:-2}"
CLIP_REFINER_HIDDEN_RATIO="${CLIP_REFINER_HIDDEN_RATIO:-0.25}"
CLIP_REFINER_RESIDUAL_SCALE="${CLIP_REFINER_RESIDUAL_SCALE:-0.25}"
CLIP_REFINER_TEMPERATURE="${CLIP_REFINER_TEMPERATURE:-0.07}"
CLIP_REFINER_LAMBDA="${CLIP_REFINER_LAMBDA:-0.25}"
CLIP_REFINER_SCD_WEIGHT="${CLIP_REFINER_SCD_WEIGHT:-0.05}"

if [[ ! -d "${VIDEO_PATH}" ]]; then
  echo "Missing LSMDC video directory: ${VIDEO_PATH}" >&2
  exit 1
fi

ANNO_FILES=(
  "${DATA_PATH}/LSMDC16_annos_training.csv"
  "${DATA_PATH}/LSMDC16_annos_val.csv"
  "${DATA_PATH}/LSMDC16_challenge_1000_publictect.csv"
)

for path in "${ANNO_FILES[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
done

validate_video_path_json() {
  LSMDC_VIDEO_PATH="${VIDEO_PATH}" \
  LSMDC_VIDEO_PATH_JSON="${VIDEO_PATH_JSON}" \
  LSMDC_ANNO_FILES="${ANNO_FILES[*]}" \
  "${PYTHON_BIN}" - <<'PY'
import json
import os
import sys
from pathlib import Path

video_dir = Path(os.environ["LSMDC_VIDEO_PATH"])
json_path = Path(os.environ["LSMDC_VIDEO_PATH_JSON"])
anno_files = [Path(item) for item in os.environ["LSMDC_ANNO_FILES"].split()]

try:
    with json_path.open("r", encoding="utf-8") as fp:
        video_dict = json.load(fp)
except Exception as exc:
    print("Cannot read {}: {}".format(json_path, exc), file=sys.stderr)
    sys.exit(1)

if not isinstance(video_dict, dict) or not video_dict:
    print("{} is empty or invalid.".format(json_path), file=sys.stderr)
    sys.exit(1)

clip_ids = []
seen = set()
for anno_file in anno_files:
    with anno_file.open("r", encoding="utf-8", errors="replace") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            clip_id = line.split("\t", 1)[0]
            if clip_id not in seen:
                seen.add(clip_id)
                clip_ids.append(clip_id)

missing_in_json = []
missing_files = []
for clip_id in clip_ids:
    rel_path = video_dict.get(clip_id)
    if rel_path is None:
        missing_in_json.append(clip_id)
        continue
    file_path = Path(rel_path)
    if not file_path.is_absolute():
        file_path = video_dir / file_path
    if not file_path.is_file():
        missing_files.append("{} -> {}".format(clip_id, rel_path))

if missing_in_json or missing_files:
    print(
        "{} does not match VIDEO_PATH={}.".format(json_path, video_dir),
        file=sys.stderr,
    )
    if missing_in_json:
        print(
            "  Missing annotated ids: {}, examples: {}".format(
                len(missing_in_json), missing_in_json[:5]
            ),
            file=sys.stderr,
        )
    if missing_files:
        print(
            "  Missing referenced files: {}, examples: {}".format(
                len(missing_files), missing_files[:5]
            ),
            file=sys.stderr,
        )
    sys.exit(1)

print(
    "Using {} with {} entries covering {} annotated LSMDC clips.".format(
        json_path, len(video_dict), len(clip_ids)
    )
)
PY
}

generate_video_path_json() {
  echo "Generating ${VIDEO_PATH_JSON} from ${VIDEO_PATH}"
  LSMDC_VIDEO_PATH="${VIDEO_PATH}" \
  LSMDC_VIDEO_PATH_JSON="${VIDEO_PATH_JSON}" \
  LSMDC_MIN_VIDEO_BYTES="${LSMDC_MIN_VIDEO_BYTES}" \
  "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

video_dir = Path(os.environ["LSMDC_VIDEO_PATH"])
output = Path(os.environ["LSMDC_VIDEO_PATH_JSON"])
min_bytes = int(os.environ.get("LSMDC_MIN_VIDEO_BYTES", "0"))
video_extensions = {
    ".avi",
    ".flv",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
    ".wmv",
}

def should_replace(old_rel, new_rel):
    old_suffix = Path(old_rel).suffix.lower()
    new_suffix = Path(new_rel).suffix.lower()
    if old_suffix != ".mp4" and new_suffix == ".mp4":
        return True
    if old_suffix == ".mp4" and new_suffix != ".mp4":
        return False
    return new_rel < old_rel

video_dict = {}
for file_path in video_dir.rglob("*"):
    if not file_path.is_file() or file_path.suffix.lower() not in video_extensions:
        continue
    try:
        if file_path.stat().st_size < min_bytes:
            continue
    except FileNotFoundError:
        continue
    video_id = file_path.stem
    rel_path = file_path.relative_to(video_dir).as_posix()
    old_rel = video_dict.get(video_id)
    if old_rel is None or should_replace(old_rel, rel_path):
        video_dict[video_id] = rel_path

if not video_dict:
    raise SystemExit("No usable video files found under {}".format(video_dir))

output.parent.mkdir(parents=True, exist_ok=True)
tmp_output = output.with_name(output.name + ".tmp")
with tmp_output.open("w", encoding="utf-8") as fp:
    json.dump(video_dict, fp, indent=4, sort_keys=True)
os.replace(str(tmp_output), str(output))

print("Saved {} video paths to {}".format(len(video_dict), output))
PY
}

case "${REBUILD_VIDEO_PATH_JSON,,}" in
  1|true|yes|force)
    generate_video_path_json
    validate_video_path_json
    ;;
  0|false|no|never)
    if [[ ! -f "${VIDEO_PATH_JSON}" ]]; then
      echo "Missing required file: ${VIDEO_PATH_JSON}" >&2
      exit 1
    fi
    ;;
  auto)
    if [[ ! -f "${VIDEO_PATH_JSON}" ]]; then
      generate_video_path_json
      validate_video_path_json
    elif ! validate_video_path_json; then
      generate_video_path_json
      validate_video_path_json
    fi
    ;;
  *)
    echo "Unsupported REBUILD_VIDEO_PATH_JSON=${REBUILD_VIDEO_PATH_JSON}; use auto, 1, or 0." >&2
    exit 1
    ;;
esac

required_files=(
  "${ANNO_FILES[@]}"
  "${VIDEO_PATH_JSON}"
  "${PRETRAINED_MODEL}"
)

if [[ -n "${INIT_MODEL}" ]]; then
  required_files+=("${INIT_MODEL}")
fi

for path in "${required_files[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
done

PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}" "${PYTHON_BIN}" - <<'PY'
import os

modules = ["torch", "torchvision", "scipy", "ftfy", "regex"]
decoder = os.environ.get("VIDEO_DECODER", "decord").lower()
if decoder in ("cv2", "opencv", "auto"):
    modules.append("cv2")
elif decoder == "decord":
    modules.append("decord")
else:
    raise SystemExit("Unsupported VIDEO_DECODER={!r}. Use decord, cv2, opencv, or auto.".format(decoder))

missing = []
for module in modules:
    try:
        __import__(module)
    except Exception as exc:
        missing.append("{} ({})".format(module, exc))
if missing:
    raise SystemExit("Missing Python package(s): {}".format(", ".join(missing)))
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
            "Use 2-2-3 for 12-frame LSMDC A0.".format(
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

init_args=()
if [[ -n "${INIT_MODEL}" ]]; then
  init_args=(--init_model "${INIT_MODEL}")
fi

echo "Training TempMe-Rf on LSMDC"
echo "  project: ${PROJECT_DIR}"
echo "  data: ${DATA_PATH}"
echo "  videos: ${VIDEO_PATH}"
echo "  video path json: ${VIDEO_PATH_JSON}"
echo "  output: ${OUTPUT_PATH}"
echo "  pretrained: ${PRETRAINED_MODEL}"
echo "  init model: ${INIT_MODEL:-None}"
echo "  decoder: ${VIDEO_DECODER}"
echo "  used gpus: ${CUDA_VISIBLE_DEVICES}"
echo "  gpus/processes: ${NPROC_PER_NODE}"
echo "  A0 merge layer: ${MERGE_LAYER}"
echo "  A0 merge frame num: ${MERGE_FRAME_NUM}"
echo "  A0 token removal: ${MERGE_TOKEN_PROPORTION}"
echo "  Rf flags: claa=${USE_CLAA} sap=${USE_SAP} pamr=${USE_PAMR} refiner=${USE_CLIP_REFINER}"

PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}" "${PYTHON_BIN}" -m torch.distributed.launch \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  main.py \
  --do_train 1 \
  --workers "${WORKERS}" \
  --anno_path "${DATA_PATH}" \
  --video_path "${VIDEO_PATH}" \
  --datatype lsmdc \
  --output_dir "${OUTPUT_PATH}" \
  --pretrained_path "${PRETRAINED_PATH}" \
  "${init_args[@]}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --batch_size_val "${BATCH_SIZE_VAL}" \
  --max_words "${MAX_WORDS}" \
  --max_frames "${MAX_FRAMES}" \
  --video_framerate "${VIDEO_FRAMERATE}" \
  --clip_lr "${CLIP_LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --warmup_proportion "${WARMUP_PROPORTION}" \
  --n_display "${N_DISPLAY}" \
  --base_encoder "${BASE_ENCODER}" \
  --lora_dim "${LORA_DIM}" \
  --tome_r "${TOME_R}" \
  --merge_layer "${MERGE_LAYER}" \
  --merge_frame_num "${MERGE_FRAME_NUM}" \
  --merge_token_proportion "${MERGE_TOKEN_PROPORTION}" \
  --frame_pos "${FRAME_POS}" \
  --dist_backend "${DIST_BACKEND}" \
  --ddp_mode "${DDP_MODE}" \
  --use_amp "${USE_AMP}" \
  --amp_dtype "${AMP_DTYPE}" \
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
