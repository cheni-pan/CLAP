# CLAP: Cross-Layer Attention Prior Tuning for Text-Video Retrieval

This repository contains the PyTorch implementation of **CLAP** (Cross-Layer Attention Prior Tuning), an efficient adaptation method for CLIP-based text-video retrieval.

**Authors:** Zhenyi Pan, Xiangpeng Li, Deqiang Ouyang  
**Code:** [cheni-pan/CLIP](https://github.com/cheni-pan/CLIP)

CLAP adapts a pretrained CLIP dual encoder with parameter-efficient components and visual attention priors:

- **SAP (Semantic Affinity Prior)** converts CLS-patch semantic affinities into a visual attention prior before softmax.
- **CLAA (Cross-Layer Attention Aggregation)** aggregates detached attention maps from preceding visual Transformer blocks and injects cross-layer structure.
- **LoRA** provides parameter-efficient adaptation of the CLIP backbone.
- **ToMe** optionally reduces redundant visual tokens and frames during the forward pass.
- **PAMR** and **CLIPRefiner** are optional implementation paths for additional spatial refinement.

The training and evaluation entry point is `main.py`. Dataset-specific launchers are provided in `scripts/`.

## Overview

The model samples video frames, encodes text and visual tokens with CLIP, and optimizes a symmetric text-to-video/video-to-text contrastive objective. During visual encoding, SAP uses CLS-conditioned patch affinities to construct a semantic prior, while CLAA fuses attention maps from preceding visual blocks. Both priors are detached during construction and can be enabled independently through the command-line options below. The shared `resclip_window` setting controls the visual-block window in which these priors are applied.

The paper evaluates CLAP on MSR-VTT, ActivityNet, DiDeMo, and LSMDC. The repository also includes optional ToMe, PAMR, and CLIPRefiner paths used by some launch presets.

The implementation reports the standard retrieval metrics in both directions:

- Text-to-video: `R@1`, `R@5`, `R@10`, `R@50`, median rank, mean rank, and `R@sum`.
- Video-to-text: the same metrics.
- A balanced retrieval score (`Balance R@sum`) is also logged.

## Requirements

The code requires Python and a PyTorch environment with the following packages:

```bash
pip install torch torchvision numpy scipy pandas pillow tqdm
```

For video decoding, install at least one supported backend:

```bash
pip install decord opencv-python
```

The data loader supports `decord`, OpenCV, `torchvision`, and `ffmpeg`. Select a decoder with `VIDEO_DECODER=decord`, `VIDEO_DECODER=cv2`, `VIDEO_DECODER=torchvision`, `VIDEO_DECODER=ffmpeg`, or `VIDEO_DECODER=auto`.

The repository does not include pretrained CLIP weights or video datasets. Download them separately and pass their paths to the launch scripts.

## Repository Structure

```text
.
|-- main.py                         # Training and evaluation entry point
|-- scripts/                        # Dataset-specific training/evaluation scripts
|   |-- MSRVTT.sh
|   |-- MSVD.sh / MSVD_infer.sh
|   |-- DiDeMo.sh / DiDeMo_infer.sh
|   |-- ActivityNet.sh / ActivityNet_infer.sh
|   |-- LSMDC.sh / LSMDC_infer.sh
|   `-- ActivityNet_sweep.sh
`-- tvr/
    |-- dataloaders/                # Dataset readers and video decoding
    |-- models/                     # CLIP, LoRA, ToMe, CLAA, SAP, PAMR
    `-- utils/                      # Distributed utilities, logging, metrics
```

## Data Preparation

Set `DATA_PATH` to the annotation directory and `VIDEO_PATH` to the directory containing the videos. The loaders expect the following files.

| Dataset | Required annotation files | Video layout |
| --- | --- | --- |
| MSR-VTT | `MSRVTT_train.9k.csv`, `MSRVTT_JSFUSION_test.csv`, `MSRVTT_data.json` | Video files named `<video_id>.mp4` |
| MSVD | `train_list.txt`, `val_list.txt`, `test_list.txt`, `raw-captions.pkl` | Files whose stem matches a video id |
| DiDeMo | `train_list.txt`, `val_list.txt`, `test_list.txt`, `train_data.json`, `val_data.json`, `test_data.json` | Files named by the ids in the split lists |
| ActivityNet | `train_ids.json`, `train.json`, `val_ids.json`, `val_1.json` | Video files whose stem matches an ActivityNet id |
| LSMDC | `LSMDC16_annos_training.csv`, `LSMDC16_annos_val.csv`, `LSMDC16_challenge_1000_publictect.csv` | A `video_path.json` id-to-relative-path mapping plus the video directory |

For LSMDC, set `VIDEO_PATH_JSON` when the mapping is outside `DATA_PATH`. The LSMDC scripts can validate or rebuild this mapping when `REBUILD_VIDEO_PATH_JSON=auto`.

## Training

Run commands from the repository root. The scripts contain dataset-specific defaults, GPU settings, merge schedules, and CLAA/SAP hyperparameters. Override any setting with an environment variable.

### MSR-VTT

```bash
DATA_PATH=/path/to/MSR-VTT/anns \
VIDEO_PATH=/path/to/MSR-VTT/videos \
PRETRAINED_PATH=/path/to/clip/checkpoints \
OUTPUT_PATH=log/MSRVTT \
NPROC_PER_NODE=4 \
bash scripts/MSRVTT.sh
```

### MSVD

```bash
DATA_PATH=/path/to/MSVD/annotations \
VIDEO_PATH=/path/to/MSVD/videos \
PRETRAINED_PATH=/path/to/clip/checkpoints \
OUTPUT_PATH=log/MSVD \
NPROC_PER_NODE=4 \
bash scripts/MSVD.sh
```

### DiDeMo

```bash
DATA_PATH=/path/to/DiDeMo/annotations \
VIDEO_PATH=/path/to/DiDeMo/videos \
PRETRAINED_PATH=/path/to/clip/checkpoints \
OUTPUT_PATH=log/DiDeMo \
NPROC_PER_NODE=4 \
bash scripts/DiDeMo.sh
```

### ActivityNet

```bash
DATA_PATH=/path/to/ActivityNet/annotations \
VIDEO_PATH=/path/to/ActivityNet/videos \
PRETRAINED_PATH=/path/to/clip/checkpoints \
OUTPUT_PATH=log/ActivityNet \
NPROC_PER_NODE=4 \
bash scripts/ActivityNet.sh
```

### LSMDC

```bash
DATA_PATH=/path/to/LSMDC \
VIDEO_PATH=/path/to/LSMDC/videos \
VIDEO_PATH_JSON=/path/to/LSMDC/video_path.json \
PRETRAINED_PATH=/path/to/clip/checkpoints \
OUTPUT_PATH=log/LSMDC \
NPROC_PER_NODE=2 \
bash scripts/LSMDC.sh
```

The lightweight wrappers `scripts/LSMDC_train.sh` and `scripts/lsmdc_t.sh` provide alternate LSMDC training presets.

## Evaluation

Evaluation scripts load a checkpoint with `--init_model` and write logs and metrics under `OUTPUT_PATH`.

```bash
CKPT_PATH=/path/to/checkpoint/best.pth \
DATA_PATH=/path/to/ActivityNet/annotations \
VIDEO_PATH=/path/to/ActivityNet/videos \
PRETRAINED_PATH=/path/to/clip/checkpoints \
OUTPUT_PATH=log/ActivityNet_infer \
NPROC_PER_NODE=1 \
bash scripts/ActivityNet_infer.sh
```

The corresponding evaluation launchers are:

```text
scripts/MSVD_infer.sh
scripts/DiDeMo_infer.sh
scripts/ActivityNet_infer.sh
scripts/LSMDC_infer.sh
```

For LSMDC, also provide `VIDEO_PATH_JSON` when needed. The evaluation split can be selected with `EVAL_SPLIT=val` or `EVAL_SPLIT=test` where supported by the dataset loader.

## Direct Python Usage

The launch scripts are recommended because they validate data files and configure distributed execution. A minimal single-process training command is:

```bash
python main.py \
  --do_train 1 \
  --datatype msrvtt \
  --anno_path /path/to/MSR-VTT/anns \
  --video_path /path/to/MSR-VTT/videos \
  --pretrained_path /path/to/clip/checkpoints \
  --output_dir log/MSRVTT \
  --device cuda \
  --use_claa 1 \
  --use_sap 1 \
  --use_pamr 0 \
  --use_clip_refiner 0
```

For evaluation, replace `--do_train 1` with `--do_eval 1` and add `--init_model /path/to/checkpoint.pth`.

## Main Configuration Options

| Option | Description | Default |
| --- | --- | --- |
| `--base_encoder` | CLIP visual backbone | `ViT-B/32` |
| `--lora_dim` | LoRA bottleneck dimension | `8` |
| `--tome_r` | Number of tokens removed by ToMe at a merge operation | `2` |
| `--merge_layer` | Visual blocks used for temporal token merging | `8-9-10` |
| `--merge_frame_num` | Frame merge schedule | `2-2-3` |
| `--merge_token_proportion` | Token removal proportions | `30-10` |
| `--use_claa` | Enable CLAA | `1` |
| `--use_sap` | Enable SAP | `1` |
| `--resclip_window` | Number of visual blocks reserved for CLAA/SAP | `4` |
| `--claa_tau`, `--claa_lambda` | CLAA scale and blend ratio | `2.0`, `0.6` |
| `--sap_tau`, `--sap_lambda` | SAP scale and blend ratio | `2.1`, `0.6` |
| `--sap_temp_thd` | Confidence threshold for SAP affinities | `0.0` |
| `--use_pamr` | Enable PAMR confidence refinement | `0` |
| `--use_clip_refiner` | Enable dense CLIP spatial refinement | `0` |
| `--max_frames` | Number of sampled video frames | `12` |
| `--max_words` | Maximum text token length | `32` |
| `--clip_lr` | Learning rate for trainable parameters | `6e-4` |

Boolean-like options are represented as integers (`0` or `1`). Dataset scripts may override the general defaults for their own frame, merge, and refinement schedules.

## Outputs and Metrics

Each run writes logs and checkpoints to `OUTPUT_PATH`. During training, the main process saves:

```text
OUTPUT_PATH/best.pth       # Best text-to-video R@1 checkpoint
OUTPUT_PATH/balance.pth    # Best balanced R@sum checkpoint
OUTPUT_PATH/log.txt        # Training and evaluation log
```

The exact files depend on the selected script and whether training or evaluation is enabled. Metrics are printed in the log after each validation/evaluation pass.

## Reproducibility

Use `--seed` to control Python, NumPy, and PyTorch seeds. For multi-GPU runs, set `CUDA_VISIBLE_DEVICES`, `NPROC_PER_NODE`, and `MASTER_PORT` explicitly. The scripts expose additional controls for distributed execution, mixed precision, and video decoding:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
MASTER_PORT=29508 \
USE_AMP=1 \
AMP_DTYPE=bf16 \
VIDEO_DECODER=decord \
bash scripts/ActivityNet.sh
```

`ActivityNet_sweep.sh` can launch the predefined ActivityNet parameter sweep. Use `RUN_SET=tier1`, `RUN_SET=tier2`, or `RUN_SET=full`; set `DRY_RUN=1` to print commands without starting training.

## Acknowledgements

This codebase builds on ideas and components from the following projects:

- [CLIP](https://github.com/openai/CLIP)
- [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip)
- [Token Merging (ToMe)](https://github.com/facebookresearch/ToMe)
- [PAMR](https://github.com/insitro/Position-Aware-Module-for-Refinement)

Please follow the licenses and citation requirements of the upstream projects when using this repository.

## Citation

If you use this code, please cite the CLAP manuscript:

```bibtex
@misc{pan2026clap,
  title   = {CLAP: Cross-Layer Attention Prior Tuning for Text-Video Retrieval},
  author  = {Pan, Zhenyi and Li, Xiangpeng and Ouyang, Deqiang},
  year    = {2026},
  note    = {Manuscript}
}
```
