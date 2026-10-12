# <i>CLAP:</i> Cross-Layer Attention Prior Tuning for Text-Video Retrieval

[Zhenyi Pan](https://github.com/cheni-pan), Xiangpeng Li, Deqiang Ouyang

[`Code`](https://github.com/cheni-pan/CLIP) | [`BibTeX`](#bibliography)

This is the official PyTorch implementation of **CLAP: Cross-Layer Attention Prior Tuning for Text-Video Retrieval**. CLAP adapts a pretrained CLIP model for efficient text-video retrieval. In this work, we:

- introduce **SAP (Semantic Affinity Prior)** to convert CLS-patch affinities into a visual attention prior;
- propose **CLAA (Cross-Layer Attention Aggregation)** to reuse attention maps from preceding visual Transformer blocks;
- combine the priors with parameter-efficient LoRA adaptation and optional token merging.

## Overview

CLAP adapts a pretrained CLIP dual encoder for text-video retrieval with parameter-efficient tuning. It constructs a semantic affinity prior from CLS-patch relations and aggregates attention across visual Transformer layers, allowing the visual encoder to preserve both token-level semantics and cross-layer structure. The model is trained with a symmetric text-to-video and video-to-text contrastive objective.

## Usage

### Requirements

```bash
pip install torch torchvision numpy scipy pandas pillow tqdm decord opencv-python
```

Download the CLIP checkpoint and the dataset annotations/videos separately.

### Datasets

We support **MSR-VTT**, **MSVD**, **DiDeMo**, **ActivityNet**, and **LSMDC**. Set the annotation, video, and pretrained-model paths in the corresponding script or override them with environment variables.

### How to Run (MSR-VTT example)

```bash
DATA_PATH=/path/to/MSR-VTT \
PRETRAINED_PATH=/path/to/clip/checkpoints \
OUTPUT_PATH=log/MSRVTT \
NPROC=4 \
bash scripts/MSRVTT.sh
```

The other training launchers are:

```text
scripts/MSVD.sh
scripts/DiDeMo.sh
scripts/ActivityNet.sh
scripts/LSMDC.sh
```

For a direct single-process run:

```bash
python main.py \
  --do_train 1 \
  --datatype msrvtt \
  --anno_path /path/to/MSR-VTT/annotations \
  --video_path /path/to/MSR-VTT/videos \
  --pretrained_path /path/to/clip/checkpoints \
  --output_dir log/MSRVTT \
  --use_claa 1 \
  --use_sap 1
```

### Evaluation

```bash
python main.py \
  --do_eval 1 \
  --datatype msrvtt \
  --anno_path /path/to/MSR-VTT/annotations \
  --video_path /path/to/MSR-VTT/videos \
  --pretrained_path /path/to/clip/checkpoints \
  --output_dir log/MSRVTT_eval \
  --init_model /path/to/checkpoint/best.pth
```

Dataset-specific inference launchers are available for MSVD, DiDeMo, ActivityNet, and LSMDC in `scripts/`.

## Bibliography

If you find this repository useful, please consider citing our work:

```bibtex
@misc{pan2026clap,
  title  = {CLAP: Cross-Layer Attention Prior Tuning for Text-Video Retrieval},
  author = {Pan, Zhenyi and Li, Xiangpeng and Ouyang, Deqiang},
  year   = {2026},
  note   = {Manuscript}
}
```

## Acknowledgements

This codebase builds on [CLIP](https://github.com/openai/CLIP), [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip), [Token Merging](https://github.com/facebookresearch/ToMe), and [PAMR](https://github.com/insitro/Position-Aware-Module-for-Refinement).
