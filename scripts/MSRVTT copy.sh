DATA_PATH="${DATA_PATH:-/mnt/data/PZY/train/MUSE-main/train/}"
OUTPUT_PATH="${OUTPUT_PATH:-log/MSRVTT}"
PRETRAINED_PATH="${PRETRAINED_PATH:-/mnt/data/PZY/train/DGL-main/DGL-main/modules/pretrained}"
INIT_MODEL="${INIT_MODEL:-/mnt/data/PZY/test/Res/TempMe-RES/log/MSRVTT/best.pth}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NPROC="${NPROC:-4}"

USE_CLAA="${USE_CLAA:-1}"
USE_SAP="${USE_SAP:-1}"
USE_PAMR="${USE_PAMR:-1}"
RESCLIP_WINDOW="${RESCLIP_WINDOW:-8}"
CLAA_TAU="${CLAA_TAU:-1.5}"
CLAA_LAMBDA="${CLAA_LAMBDA:-0.4}"
SAP_TAU="${SAP_TAU:-1.2}"
SAP_LAMBDA="${SAP_LAMBDA:-0.3}"
SAP_TEMP_THD="${SAP_TEMP_THD:-0.0}"
PAMR_STEPS="${PAMR_STEPS:-10}"
PAMR_STRIDE="${PAMR_STRIDE:-8-16}"
USE_CLIP_REFINER="${USE_CLIP_REFINER:-1}"
CLIP_REFINER_DEPTH="${CLIP_REFINER_DEPTH:-2}"
CLIP_REFINER_HIDDEN_RATIO="${CLIP_REFINER_HIDDEN_RATIO:-0.25}"
CLIP_REFINER_RESIDUAL_SCALE="${CLIP_REFINER_RESIDUAL_SCALE:-0.25}"
CLIP_REFINER_TEMPERATURE="${CLIP_REFINER_TEMPERATURE:-0.07}"
CLIP_REFINER_LAMBDA="${CLIP_REFINER_LAMBDA:-0.25}"
CLIP_REFINER_SCD_WEIGHT="${CLIP_REFINER_SCD_WEIGHT:-0.05}"

${PYTHON_BIN} -m torch.distributed.launch --nproc_per_node=${NPROC} main.py \
  --do_eval 1 --workers 8 \
  --anno_path ${DATA_PATH} --video_path ${DATA_PATH}/video --datatype msrvtt \
  --output_dir ${OUTPUT_PATH} \
  --pretrained_path ${PRETRAINED_PATH} \
  --init_model ${INIT_MODEL} \
  --use_claa ${USE_CLAA} \
  --use_sap ${USE_SAP} \
  --use_pamr ${USE_PAMR} \
  --resclip_window ${RESCLIP_WINDOW} \
  --claa_tau ${CLAA_TAU} \
  --claa_lambda ${CLAA_LAMBDA} \
  --sap_tau ${SAP_TAU} \
  --sap_lambda ${SAP_LAMBDA} \
  --sap_temp_thd ${SAP_TEMP_THD} \
  --pamr_steps ${PAMR_STEPS} \
  --pamr_stride ${PAMR_STRIDE} \
  --use_clip_refiner ${USE_CLIP_REFINER} \
  --clip_refiner_depth ${CLIP_REFINER_DEPTH} \
  --clip_refiner_hidden_ratio ${CLIP_REFINER_HIDDEN_RATIO} \
  --clip_refiner_residual_scale ${CLIP_REFINER_RESIDUAL_SCALE} \
  --clip_refiner_temperature ${CLIP_REFINER_TEMPERATURE} \
  --clip_refiner_lambda ${CLIP_REFINER_LAMBDA} \
  --clip_refiner_scd_weight ${CLIP_REFINER_SCD_WEIGHT}
 
