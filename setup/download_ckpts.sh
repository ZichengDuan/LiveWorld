#!/bin/bash
# Download all required pretrained weights into ckpts/
set -e
cd "$(dirname "$0")/.."

echo "============================================================"
echo "Downloading LiveWorld model weights"
echo "============================================================"

# LiveWorld weights (State Adapter + LoRA)
echo ">>> LiveWorld State Adapter + LoRA"
hf download ZichengD/LiveWorld ckpts/state_adapter/model.pt --local-dir .
hf download ZichengD/LiveWorld ckpts/lora/model.pt --local-dir .

# Wan2.1 T2V 14B backbone
echo ">>> Wan2.1 T2V 14B"
hf download Wan-AI/Wan2.1-T2V-14B --local-dir ckpts/Wan-AI--Wan2.1-T2V-14B

# Wan2.1 VAE (for data preparation)
echo ">>> Wan2.1 VAE"
hf download alibaba-pai/Wan2.1-Fun-1.3B-InP --local-dir ckpts/alibaba-pai--Wan2.1-Fun-1.3B-InP

# Wan2.1 distilled backbone (for fast inference)
echo ">>> Wan2.1 Distilled Backbone"
hf download lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill --local-dir ckpts/Wan2.1-T2V-14B-StepDistill

# Qwen3-VL 8B (entity detection)
echo ">>> Qwen3-VL 8B"
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir ckpts/Qwen--Qwen3-VL-8B-Instruct

# SAM3 (video segmentation)
echo ">>> SAM3"
hf download facebook/sam3 --local-dir ckpts/facebook--sam3

# Stream3R (3D reconstruction)
echo ">>> Stream3R"
hf download yslan/STream3R --local-dir ckpts/yslan--STream3R

# DINOv3 (entity matching, optional)
echo ">>> DINOv3 (optional)"
hf download facebook/dinov3-vith16plus-pretrain-lvd1689m --local-dir ckpts/facebook--dinov3-vith16plus-pretrain-lvd1689m

# Example data (inference sample + training sample)
echo ">>> Example data"
hf download ZichengD/LiveWorld --include "examples/*" --local-dir .

echo ""
echo "============================================================"
echo "All weights and example data downloaded."
echo "============================================================"
