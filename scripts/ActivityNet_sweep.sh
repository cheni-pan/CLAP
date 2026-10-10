#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
BASE_SCRIPT="${BASE_SCRIPT:-${SCRIPT_DIR}/ActivityNet.sh}"
SWEEP_ROOT="${SWEEP_ROOT:-${PROJECT_DIR}/log/ActivityNet_sweep}"

# Current local best from /mnt/data/Panzhenyi/train/Rf/log/ActivityNet/log.txt:
# R@1 44.6298, Balance R@sum 410.6998
# The sweep keeps the strong recipe fixed and only explores nearby knobs.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,1,0,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export BATCH_SIZE="${BATCH_SIZE:-128}"
export BATCH_SIZE_VAL="${BATCH_SIZE_VAL:-128}"
export MAX_WORDS="${MAX_WORDS:-64}"
export MAX_FRAMES="${MAX_FRAMES:-64}"
export MERGE_LAYER="${MERGE_LAYER:-8-9-10}"
export MERGE_FRAME_NUM="${MERGE_FRAME_NUM:-4-4-4}"
export USE_PAMR="${USE_PAMR:-0}"
export USE_CLIP_REFINER="${USE_CLIP_REFINER:-0}"
export DDP_MODE="${DDP_MODE:-manual}"
export RF_DIST_BACKEND="${RF_DIST_BACKEND:-gloo}"
export VIDEO_DECODER="${VIDEO_DECODER:-cv2}"
export USE_AMP="${USE_AMP:-1}"
export AMP_DTYPE="${AMP_DTYPE:-bf16}"

RUN_SET="${RUN_SET:-tier1}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_FINISHED="${SKIP_FINISHED:-1}"
START_INDEX="${START_INDEX:-1}"
MAX_RUNS="${MAX_RUNS:-0}"
BASE_MASTER_PORT="${BASE_MASTER_PORT:-29600}"

case "${RUN_SET}" in
  tier1|tier2|full) ;;
  *)
    echo "Unsupported RUN_SET=${RUN_SET}. Use tier1, tier2, or full." >&2
    exit 1
    ;;
esac

common_env=(
  "BASE_ENCODER=ViT-B/32"
  "LORA_DIM=8"
  "TOME_R=2"
  "USE_CLAA=0"
  "USE_SAP=1"
  "USE_PAMR=0"
  "USE_CLIP_REFINER=0"
  "MERGE_TOKEN_PROPORTION=30-10"
  "CLIP_LR=6e-4"
  "SAP_TAU=2.1"
  "SAP_LAMBDA=0.7"
  "SAP_TEMP_THD=0.0"
  "CLAA_TAU=2.0"
  "CLAA_LAMBDA=0.4"
)

tier1_runs=(
  "best_current|"
  "sap_lambda_060|SAP_LAMBDA=0.6"
  "sap_lambda_080|SAP_LAMBDA=0.8"
  "sap_tau_180|SAP_TAU=1.8"
  "sap_tau_240|SAP_TAU=2.4"
  "lr_4e-4|CLIP_LR=4e-4"
  "lr_8e-4|CLIP_LR=8e-4"
)

tier2_runs=(
  "token_25-10|MERGE_TOKEN_PROPORTION=25-10"
  "token_35-10|MERGE_TOKEN_PROPORTION=35-10"
  "token_30-15|MERGE_TOKEN_PROPORTION=30-15"
  "sap_thd_001|SAP_TEMP_THD=0.01"
  "sap_thd_003|SAP_TEMP_THD=0.03"
  "lora_dim_16|LORA_DIM=16"
  "tome_r_1|TOME_R=1"
)

full_runs=(
  "claa_only|USE_CLAA=1 USE_SAP=0 CLAA_LAMBDA=0.5"
  "claa_sap_lam040|USE_CLAA=1 USE_SAP=1 CLAA_LAMBDA=0.5 SAP_LAMBDA=0.4 SAP_TAU=1.35"
  "frames32_merge442|MAX_FRAMES=32 MERGE_FRAME_NUM=4-4-2"
  "frames48_merge443|MAX_FRAMES=48 MERGE_FRAME_NUM=4-4-3"
  "lr_5e-4_sap080|CLIP_LR=5e-4 SAP_LAMBDA=0.8"
  "lr_7e-4_sap060|CLIP_LR=7e-4 SAP_LAMBDA=0.6"
)

runs=("${tier1_runs[@]}")
if [[ "${RUN_SET}" == "tier2" || "${RUN_SET}" == "full" ]]; then
  runs+=("${tier2_runs[@]}")
fi
if [[ "${RUN_SET}" == "full" ]]; then
  runs+=("${full_runs[@]}")
fi

mkdir -p "${SWEEP_ROOT}"

summarize_results() {
  python3 - "${SWEEP_ROOT}" <<'PY'
import os
import re
import sys

root = sys.argv[1]
r1_re = re.compile(r"The best R1 is: ([0-9.]+) at all")
bal_re = re.compile(r"The best balance R@sum is: ([0-9.]+) at all")
rows = []
for name in sorted(os.listdir(root)):
    log_path = os.path.join(root, name, "log.txt")
    if not os.path.isfile(log_path):
        continue
    best_r1 = None
    best_balance = None
    with open(log_path, errors="replace") as fp:
        for line in fp:
            match = r1_re.search(line)
            if match:
                best_r1 = float(match.group(1))
            match = bal_re.search(line)
            if match:
                best_balance = float(match.group(1))
    if best_r1 is not None or best_balance is not None:
        rows.append((best_r1 or -1.0, best_balance or -1.0, name))

rows.sort(reverse=True)
print("==== ActivityNet sweep summary ====")
if not rows:
    print("No completed metrics found yet.")
else:
    for best_r1, best_balance, name in rows:
        r1_text = "NA" if best_r1 < 0 else f"{best_r1:.4f}"
        bal_text = "NA" if best_balance < 0 else f"{best_balance:.4f}"
        print(f"{name:24s} R1={r1_text:>8s} Balance={bal_text:>9s}")
PY
}

echo "ActivityNet parameter sweep"
echo "  script: ${BASE_SCRIPT}"
echo "  output: ${SWEEP_ROOT}"
echo "  run set: ${RUN_SET}"
echo "  gpus: ${CUDA_VISIBLE_DEVICES}"
echo "  processes: ${NPROC_PER_NODE}"
echo "  batch: ${BATCH_SIZE} / val ${BATCH_SIZE_VAL}"
echo "  dry run: ${DRY_RUN}"

completed_runs=0
for idx in "${!runs[@]}"; do
  run_no=$((idx + 1))
  if (( run_no < START_INDEX )); then
    continue
  fi
  if (( MAX_RUNS > 0 && completed_runs >= MAX_RUNS )); then
    break
  fi

  entry="${runs[$idx]}"
  name="${entry%%|*}"
  override_string="${entry#*|}"
  output_dir="${SWEEP_ROOT}/${run_no}_${name}"

  if [[ "${SKIP_FINISHED}" == "1" && -f "${output_dir}/log.txt" ]] && grep -q "training finished" "${output_dir}/log.txt"; then
    echo "[$run_no/${#runs[@]}] skip finished: ${name}"
    continue
  fi

  env_args=(
    "OUTPUT_PATH=${output_dir}"
    "MASTER_PORT=$((BASE_MASTER_PORT + run_no))"
    "${common_env[@]}"
  )
  if [[ -n "${override_string}" ]]; then
    read -r -a overrides <<< "${override_string}"
    env_args+=("${overrides[@]}")
  fi

  echo "[$run_no/${#runs[@]}] ${name}"
  printf '  env'
  for item in "${env_args[@]}"; do
    printf ' %q' "${item}"
  done
  printf ' bash %q' "${BASE_SCRIPT}"
  for item in "$@"; do
    printf ' %q' "${item}"
  done
  printf '\n'

  if [[ "${DRY_RUN}" == "1" ]]; then
    completed_runs=$((completed_runs + 1))
    continue
  fi

  env "${env_args[@]}" bash "${BASE_SCRIPT}" "$@"
  completed_runs=$((completed_runs + 1))
  summarize_results
done

summarize_results
