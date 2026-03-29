#!/bin/bash
# LiveWorld Inference
export CUDA_VISIBLE_DEVICES=0

python scripts/infer.py \
    --config examples/inference_sample/kid_coffee/infer_scripts/traj_f.yaml \
    --system-config configs/infer_system_config_14B.yaml \
    --output-root outputs \
    --device cuda:0
