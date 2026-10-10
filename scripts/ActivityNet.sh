#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATA_PATH="${DATA_PATH:-/mnt/data/Panzhenyi/train/data/activity_mp4}"
VIDEO_PATH="${VIDEO_PATH:-${DATA_PATH}/Activity_Videos}"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_DIR}/log/ActivityNet}"
PRETRAINED_PATH="${PRETRAINED_PATH:-/mnt/data/Panzhenyi/train/model}"
INIT_MODEL="${INIT_MODEL:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TEMPME_STARTUP_TRACE="${TEMPME_STARTUP_TRACE:-1}"
export DECORD_LOG_LEVEL="${DECORD_LOG_LEVEL:-ERROR}"
export VIDEO_DECODER="${VIDEO_DECODER:-cv2}"
export EVAL_LOG_INTERVAL="${EVAL_LOG_INTERVAL:-50}"
export DETECT_ANOMALY="${DETECT_ANOMALY:-0}"
export USE_AMP="${USE_AMP:-1}"
export AMP_DTYPE="${AMP_DTYPE:-bf16}"
export DDP_MODE="${DDP_MODE:-manual}"
if [[ -n "${DIST_BACKEND+x}" && -z "${RF_DIST_BACKEND+x}" ]]; then
  echo "Ignoring inherited DIST_BACKEND=${DIST_BACKEND}; use RF_DIST_BACKEND=nccl only if you really want NCCL." >&2
fi
DIST_BACKEND="${RF_DIST_BACKEND:-gloo}"
export DIST_BACKEND
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export DIST_TIMEOUT_SECONDS="${DIST_TIMEOUT_SECONDS:-600}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,1,0,3}"
export GPU_MIN_FREE_MB="${GPU_MIN_FREE_MB:-36000}"
export GPU_ALLOW_BUSY="${GPU_ALLOW_BUSY:-0}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29508}"

WORKERS="${WORKERS:-8}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-128}"
BATCH_SIZE_VAL="${BATCH_SIZE_VAL:-128}"
MAX_WORDS="${MAX_WORDS:-64}"
MAX_FRAMES="${MAX_FRAMES:-64}"
VIDEO_FRAMERATE="${VIDEO_FRAMERATE:-1}"
CLIP_LR="${CLIP_LR:-6e-4}"
N_DISPLAY="${N_DISPLAY:-50}"

case "${VIDEO_DECODER}" in
  decord|cv2|opencv|auto) ;;
  *)
    echo "Unsupported VIDEO_DECODER: ${VIDEO_DECODER}. Use decord, cv2, opencv, or auto." >&2
    exit 1
    ;;
esac

case "${DDP_MODE}" in
  manual|ddp) ;;
  *)
    echo "Unsupported DDP_MODE: ${DDP_MODE}. Use manual or ddp." >&2
    exit 1
    ;;
esac

case "${DIST_BACKEND}" in
  gloo|nccl) ;;
  *)
    echo "Unsupported DIST_BACKEND: ${DIST_BACKEND}. Use gloo or nccl." >&2
    exit 1
    ;;
esac

BASE_ENCODER="${BASE_ENCODER:-ViT-B/32}"
LORA_DIM="${LORA_DIM:-8}"
TOME_R="${TOME_R:-2}"
MERGE_LAYER="${MERGE_LAYER:-8-9-10}"
MERGE_FRAME_NUM="${MERGE_FRAME_NUM:-4-4-4}"
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

USE_CLAA="${USE_CLAA:-1}"
USE_SAP="${USE_SAP:-1}"
USE_PAMR="${USE_PAMR:-0}"
RESCLIP_WINDOW="${RESCLIP_WINDOW:-8}"
CLAA_TAU="${CLAA_TAU:-2.0}"
CLAA_LAMBDA="${CLAA_LAMBDA:-0.15}"
SAP_TAU="${SAP_TAU:-1.5}"
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

required_files=(
  "${DATA_PATH}/train_ids.json"
  "${DATA_PATH}/train.json"
  "${DATA_PATH}/val_ids.json"
  "${DATA_PATH}/val_1.json"
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

if [[ ! -d "${VIDEO_PATH}" ]]; then
  echo "Missing ActivityNet video directory: ${VIDEO_PATH}" >&2
  exit 1
fi

"${PYTHON_BIN}" - <<'PY'
missing = []
for module in ("cv2", "decord", "torchvision"):
    try:
        __import__(module)
    except Exception:
        missing.append(module)
if missing:
    raise SystemExit(
        "Missing Python package(s): {}. Install them in the active environment before training.".format(
            ", ".join(missing)
        )
    )
PY

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" NPROC_PER_NODE="${NPROC_PER_NODE}" "${PYTHON_BIN}" - <<'PY'
import os
import torch

try:
    requested_processes = int(os.environ["NPROC_PER_NODE"])
except ValueError as exc:
    raise SystemExit("NPROC_PER_NODE must be an integer.") from exc

if requested_processes < 1:
    raise SystemExit("NPROC_PER_NODE must be at least 1.")

visible_devices = torch.cuda.device_count()
if visible_devices < requested_processes:
    raise SystemExit(
        "NPROC_PER_NODE={} but only {} CUDA device(s) are visible after "
        "CUDA_VISIBLE_DEVICES={!r}. Use valid GPU IDs or lower NPROC_PER_NODE.".format(
            requested_processes,
            visible_devices,
            os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        )
    )
PY

if [[ "${GPU_ALLOW_BUSY:-0}" != "1" && "${GPU_MIN_FREE_MB:-36000}" != "0" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" GPU_MIN_FREE_MB="${GPU_MIN_FREE_MB}" "${PYTHON_BIN}" - <<'PY'
import os
import subprocess
import sys

min_free_mb = int(os.environ.get("GPU_MIN_FREE_MB", "0"))
visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")

if min_free_mb <= 0:
    raise SystemExit(0)

entries = [item.strip() for item in visible.split(",") if item.strip()]
if not entries or visible.lower() == "all":
    entries = []

numeric_entries = [item for item in entries if item.isdigit()]
if entries and len(numeric_entries) != len(entries):
    print(
        "GPU memory guard skipped: CUDA_VISIBLE_DEVICES={!r} is not a plain numeric GPU list.".format(visible),
        file=sys.stderr,
    )
    raise SystemExit(0)

try:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
except Exception as exc:
    print("GPU memory guard skipped: nvidia-smi query failed: {}".format(exc), file=sys.stderr)
    raise SystemExit(0)

gpu_mem = {}
for line in output.splitlines():
    if not line.strip():
        continue
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 4:
        continue
    index, total_mb, used_mb, free_mb = parts[:4]
    gpu_mem[index] = (int(total_mb), int(used_mb), int(free_mb))

gpu_indices = numeric_entries if numeric_entries else sorted(gpu_mem, key=int)
busy = []
for index in gpu_indices:
    if index not in gpu_mem:
        busy.append("GPU {}: not found by nvidia-smi".format(index))
        continue
    total_mb, used_mb, free_mb = gpu_mem[index]
    if free_mb < min_free_mb:
        busy.append(
            "GPU {}: free {} MiB < required {} MiB (used {} / total {} MiB)".format(
                index, free_mb, min_free_mb, used_mb, total_mb
            )
        )

if busy:
    print("GPU memory guard failed before launch:", file=sys.stderr)
    for item in busy:
        print("  {}".format(item), file=sys.stderr)
    try:
        apps = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_memory,process_name",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    except Exception:
        apps = ""
    if apps:
        print("Current compute processes:", file=sys.stderr)
        print(apps, file=sys.stderr)
    print("Set GPU_ALLOW_BUSY=1 to bypass this guard, or choose idle GPUs with CUDA_VISIBLE_DEVICES.", file=sys.stderr)
    raise SystemExit(1)

print(
    "GPU memory guard passed: GPUs {} all have at least {} MiB free.".format(
        ",".join(gpu_indices), min_free_mb
    )
)
PY
fi

BATCH_SIZE="${BATCH_SIZE}" BATCH_SIZE_VAL="${BATCH_SIZE_VAL}" NPROC_PER_NODE="${NPROC_PER_NODE}" "${PYTHON_BIN}" - <<'PY'
import os

world_size = int(os.environ["NPROC_PER_NODE"])
batch_size = int(os.environ["BATCH_SIZE"])
batch_size_val = int(os.environ["BATCH_SIZE_VAL"])

if batch_size % world_size != 0 or batch_size_val % world_size != 0:
    raise SystemExit(
        "BATCH_SIZE and BATCH_SIZE_VAL must be divisible by NPROC_PER_NODE: "
        "BATCH_SIZE={} BATCH_SIZE_VAL={} NPROC_PER_NODE={}".format(
            batch_size,
            batch_size_val,
            world_size,
        )
    )
PY

PRETRAINED_MODEL="${PRETRAINED_MODEL}" "${PYTHON_BIN}" - <<'PY'
import os
import tempfile
import torch

model_path = os.environ["PRETRAINED_MODEL"]
cache_path = "{}.state_dict.pt".format(model_path)

cache_is_fresh = (
    os.path.exists(cache_path)
    and os.path.getmtime(cache_path) >= os.path.getmtime(model_path)
)
if cache_is_fresh:
    print("Using cached CLIP state_dict: {}".format(cache_path))
else:
    print("Preparing CLIP state_dict cache: {}".format(cache_path), flush=True)
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = model.state_dict()
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    cache_dir = os.path.dirname(cache_path) or "."
    fd, tmp_path = tempfile.mkstemp(
        prefix=".{}.".format(os.path.basename(cache_path)),
        suffix=".tmp",
        dir=cache_dir,
    )
    os.close(fd)
    try:
        torch.save(state_dict, tmp_path)
        os.replace(tmp_path, cache_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
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
            "Invalid MERGE_FRAME_NUM={} for MAX_FRAMES={}: {} cannot be divided by {}.".format(
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

echo "Training TempMe-Rf on ActivityNet"
echo "  project: ${PROJECT_DIR}"
echo "  data: ${DATA_PATH}"
echo "  videos: ${VIDEO_PATH}"
echo "  output: ${OUTPUT_PATH}"
echo "  pretrained: ${PRETRAINED_MODEL}"
echo "  init model: ${INIT_MODEL:-None}"
echo "  used gpus: ${CUDA_VISIBLE_DEVICES}"
echo "  gpus/processes: ${NPROC_PER_NODE}"
echo "  gpu min free: ${GPU_MIN_FREE_MB} MiB (allow busy: ${GPU_ALLOW_BUSY})"
echo "  video decoder: ${VIDEO_DECODER}"
echo "  ddp mode: ${DDP_MODE}"
echo "  dist backend: ${DIST_BACKEND}"
echo "  amp: ${USE_AMP} (${AMP_DTYPE})"
echo "  merge layer: ${MERGE_LAYER}"
echo "  merge frame num: ${MERGE_FRAME_NUM}"
echo "  token removal: ${MERGE_TOKEN_PROPORTION}"
echo "  Rf flags: claa=${USE_CLAA} sap=${USE_SAP} pamr=${USE_PAMR} refiner=${USE_CLIP_REFINER}"

train_args=(
  "${PROJECT_DIR}/main.py"
  --do_train 1
  --workers "${WORKERS}"
  --anno_path "${DATA_PATH}"
  --video_path "${VIDEO_PATH}"
  --datatype activity
  --output_dir "${OUTPUT_PATH}"
  --pretrained_path "${PRETRAINED_PATH}"
  "${init_args[@]}"
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --batch_size_val "${BATCH_SIZE_VAL}"
  --max_words "${MAX_WORDS}"
  --max_frames "${MAX_FRAMES}"
  --video_framerate "${VIDEO_FRAMERATE}"
  --clip_lr "${CLIP_LR}"
  --n_display "${N_DISPLAY}"
  --base_encoder "${BASE_ENCODER}"
  --lora_dim "${LORA_DIM}"
  --tome_r "${TOME_R}"
  --merge_layer "${MERGE_LAYER}"
  --merge_frame_num "${MERGE_FRAME_NUM}"
  --merge_token_proportion "${MERGE_TOKEN_PROPORTION}"
  --frame_pos "${FRAME_POS}"
  --use_claa "${USE_CLAA}"
  --use_sap "${USE_SAP}"
  --use_pamr "${USE_PAMR}"
  --resclip_window "${RESCLIP_WINDOW}"
  --claa_tau "${CLAA_TAU}"
  --claa_lambda "${CLAA_LAMBDA}"
  --sap_tau "${SAP_TAU}"
  --sap_lambda "${SAP_LAMBDA}"
  --sap_temp_thd "${SAP_TEMP_THD}"
  --pamr_steps "${PAMR_STEPS}"
  --pamr_stride "${PAMR_STRIDE}"
  --use_clip_refiner "${USE_CLIP_REFINER}"
  --clip_refiner_depth "${CLIP_REFINER_DEPTH}"
  --clip_refiner_hidden_ratio "${CLIP_REFINER_HIDDEN_RATIO}"
  --clip_refiner_residual_scale "${CLIP_REFINER_RESIDUAL_SCALE}"
  --clip_refiner_temperature "${CLIP_REFINER_TEMPERATURE}"
  --clip_refiner_lambda "${CLIP_REFINER_LAMBDA}"
  --clip_refiner_scd_weight "${CLIP_REFINER_SCD_WEIGHT}"
  --ddp_mode "${DDP_MODE}"
  --dist_backend "${DIST_BACKEND}"
  --detect_anomaly "${DETECT_ANOMALY}"
  --use_amp "${USE_AMP}"
  --amp_dtype "${AMP_DTYPE}"
  "$@"
)

if [[ "${NPROC_PER_NODE}" -eq 1 ]]; then
  launch_cmd=("${PYTHON_BIN}" "${train_args[@]}")
else
  launch_cmd=(
    "${PYTHON_BIN}" -m torch.distributed.run
    "--nproc_per_node=${NPROC_PER_NODE}"
    "--master_port=${MASTER_PORT}"
    "${train_args[@]}"
  )
fi

"${launch_cmd[@]}"
