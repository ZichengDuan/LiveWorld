#!/bin/bash
# LiveWorld Inference Sample Creation
# Generates inference configs (trajectory + storyline) from source images.
set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================
IMAGES_DIR="examples/inference_sample/raw"
OUT_ROOT="examples/inference_sample/processed"

QWEN_MODEL_PATH="ckpts/Qwen--Qwen3-VL-8B-Instruct"
SAM3_MODEL_PATH="ckpts/facebook--sam3/sam3.pt"
STREAM3R_MODEL_PATH="ckpts/yslan--STream3R"
SYSTEM_CONFIG="configs/infer_system_config_14B.yaml"

SOURCE_FPR=65
TARGET_FPR=65
MAX_STORYLINE_STEPS=4
TRAJ_CASE_TYPES="case1,case2"
SCENE_ESTIMATOR="stream3r"
TARGET_H=480
TARGET_W=832
SCENE_VOXEL_SIZE=0.01
TRAJ_DISTANCE_MODE="screen_uniform"
TRAJ_TARGET_SHIFT_RATIO=0.18
TRAJ_SCREEN_DEPTH_PERCENTILE=45
TRAJ_DISTANCE_BLEND_ALPHA=0.7
TRAJ_EXIT_VISIBLE_RATIO=0.2
SAVE_TRAJECTORY_OVERLAY=1
RESUME=1
OVERWRITE=0

# ============================================================================
# Node config
# ============================================================================
NODES=("$(hostname)")
CUDA_VISIBLE_DEVICES_LIST=("1")

# ============================================================================
# Auto-detect current node
# ============================================================================
HOSTNAME=$(hostname)
NODE_RANK=""

for i in "${!NODES[@]}"; do
  if [[ "$HOSTNAME" == "${NODES[$i]}" ]]; then
    NODE_RANK=$i
    break
  fi
done

if [[ -z "$NODE_RANK" ]]; then
  echo "[error] hostname ($HOSTNAME) not in NODES list: ${NODES[*]}"
  exit 1
fi

WORLD_SIZE=0
for gpus in "${CUDA_VISIBLE_DEVICES_LIST[@]}"; do
  n=$(echo "$gpus" | awk -F',' '{print NF}')
  WORLD_SIZE=$((WORLD_SIZE + n))
done

START_RANK=0
for ((i=0; i<NODE_RANK; i++)); do
  n=$(echo "${CUDA_VISIBLE_DEVICES_LIST[$i]}" | awk -F',' '{print NF}')
  START_RANK=$((START_RANK + n))
done

IFS=',' read -ra LOCAL_GPU_IDS <<< "${CUDA_VISIBLE_DEVICES_LIST[$NODE_RANK]}"
LOCAL_GPUS=${#LOCAL_GPU_IDS[@]}

echo "============================================================"
echo "LiveWorld Inference Sample Creation"
echo "============================================================"
echo "Node: $HOSTNAME (rank $NODE_RANK)"
echo "Local GPUs: ${LOCAL_GPU_IDS[*]} ($LOCAL_GPUS)"
echo "Global: start_rank=$START_RANK, world_size=$WORLD_SIZE"
echo "IMAGES_DIR: $IMAGES_DIR"
echo "OUT_ROOT: $OUT_ROOT"
echo "============================================================"
echo ""

RESUME_FLAG="--resume"
[[ "${RESUME}" == "0" ]] && RESUME_FLAG="--no-resume"
EXTRA_ARGS=("$RESUME_FLAG")
[[ "${OVERWRITE}" == "1" ]] && EXTRA_ARGS+=("--overwrite")
[[ "${SAVE_TRAJECTORY_OVERLAY}" == "0" ]] && EXTRA_ARGS+=("--no-save-trajectory-overlay")

# ============================================================================
# Collect all images
# ============================================================================
ALL_IMAGES=()
while IFS= read -r -d '' f; do
  ALL_IMAGES+=("$f")
done < <(find "$IMAGES_DIR" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' -o -iname '*.bmp' \) -print0 | sort -z)

TOTAL_IMAGES=${#ALL_IMAGES[@]}
if [[ $TOTAL_IMAGES -eq 0 ]]; then
  echo "[error] No images found under $IMAGES_DIR"
  exit 1
fi
echo "[assemble] found $TOTAL_IMAGES images total"

# ============================================================================
# Split images across GPUs and launch workers
# ============================================================================
PIDS=()

for local_idx in "${!LOCAL_GPU_IDS[@]}"; do
  gpu_id=${LOCAL_GPU_IDS[$local_idx]}
  global_rank=$((START_RANK + local_idx))

  TMPFILE=$(mktemp /tmp/create_infer_sample_gpu${gpu_id}.XXXXXX)
  COUNT=0
  for ((c=global_rank; c<TOTAL_IMAGES; c+=WORLD_SIZE)); do
    echo "${ALL_IMAGES[$c]}" >> "$TMPFILE"
    COUNT=$((COUNT + 1))
  done

  if [[ $COUNT -eq 0 ]]; then
    echo "[assemble] GPU $gpu_id (rank $global_rank/$WORLD_SIZE): 0 images, skip"
    rm -f "$TMPFILE"
    continue
  fi

  echo "[assemble] GPU $gpu_id (rank $global_rank/$WORLD_SIZE): $COUNT images"

  CUDA_VISIBLE_DEVICES=$gpu_id \
  python scripts/create_infer_sample/assemble_event_bench.py \
    --trajectory-case-types "$TRAJ_CASE_TYPES" \
    --images-list "$TMPFILE" \
    --output-root "$OUT_ROOT" \
    --source-frames-per-round "$SOURCE_FPR" \
    --target-frames-per-iter "$TARGET_FPR" \
    --qwen-model-path "$QWEN_MODEL_PATH" \
    --sam3-model-path "$SAM3_MODEL_PATH" \
    --scene-estimator "$SCENE_ESTIMATOR" \
    --trajectory-distance-mode "$TRAJ_DISTANCE_MODE" \
    --trajectory-target-shift-ratio "$TRAJ_TARGET_SHIFT_RATIO" \
    --trajectory-screen-depth-percentile "$TRAJ_SCREEN_DEPTH_PERCENTILE" \
    --trajectory-distance-blend-alpha "$TRAJ_DISTANCE_BLEND_ALPHA" \
    --trajectory-exit-visible-ratio "$TRAJ_EXIT_VISIBLE_RATIO" \
    --stream3r-model-path "$STREAM3R_MODEL_PATH" \
    --target-height "$TARGET_H" \
    --target-width "$TARGET_W" \
    --scene-voxel-size "$SCENE_VOXEL_SIZE" \
    --device "cuda:0" \
    --system-config-path "$SYSTEM_CONFIG" \
    --max-storyline-steps "$MAX_STORYLINE_STEPS" \
    "${EXTRA_ARGS[@]}" &

  PIDS+=($!)
done

echo ""
echo "[assemble] Launched ${#PIDS[@]} processes. PIDs: ${PIDS[*]}"
echo ""

wait

echo "[done] All processes on $HOSTNAME completed."
