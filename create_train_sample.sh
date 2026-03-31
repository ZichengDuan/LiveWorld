#!/bin/bash
# LiveWorld Data Preparation
# Edit configs/data_preparation.yaml for all settings.

export HF_HUB_DISABLE_IMPORTS_VERIFICATION=1
export NCCL_TIMEOUT=1800
export SAM3_TQDM_DISABLE=1

# ============================================================================
# Node config — define nodes and per-node GPU assignments
# ============================================================================
NODES=("$(hostname)")
CUDA_VISIBLE_DEVICES_LIST=("1")

# ============================================================================
# Auto-detect current node
# ============================================================================
HOSTNAME=$(hostname)
for i in "${!NODES[@]}"; do
    if [[ "$HOSTNAME" == "${NODES[$i]}" ]]; then
        NODE_RANK=$i
        export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_LIST[$i]}
        break
    fi
done

if [[ -z "$NODE_RANK" ]]; then
    echo "Error: hostname ($HOSTNAME) not in NODES list."
    exit 1
fi

NNODES=${#NODES[@]}

# Compute global world_size and this node's start rank
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

IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"

echo "============================================================"
echo "LiveWorld Data Preparation"
echo "============================================================"
echo "Node: $HOSTNAME | rank=$NODE_RANK | GPUs=${GPU_IDS[*]} | world_size=$WORLD_SIZE"
echo "============================================================"
echo ""

# ============================================================================
# Helper: launch one Python module per GPU in parallel, then wait
# ============================================================================
launch_per_gpu() {
    local module=$1
    shift
    local PIDS=()

    for local_rank in "${!GPU_IDS[@]}"; do
        gpu_id=${GPU_IDS[$local_rank]}
        global_rank=$((START_RANK + local_rank))
        echo "  rank $global_rank/$WORLD_SIZE on GPU $gpu_id"

        RANK=$global_rank \
        WORLD_SIZE=$WORLD_SIZE \
        LOCAL_RANK=0 \
        CUDA_VISIBLE_DEVICES=$gpu_id \
        python -m "$module" "$@" &

        PIDS+=($!)
    done

    wait "${PIDS[@]}"
}

# ============================================================================
# Step 1: Extract clips + detect entities + segment + estimate geometry + build samples
# ============================================================================
echo ">>> Step 1/4: Pipeline (clips + entities + geometry + samples)"
launch_per_gpu scripts.create_train_data.step1_build_samples

# ============================================================================
# Step 2: Generate video captions
# ============================================================================
echo ""
echo ">>> Step 2/4: Video captioning"
launch_per_gpu scripts.create_train_data.step2_captioning

# ============================================================================
# Step 3: VAE-encode videos to latents
# ============================================================================
echo ""
echo ">>> Step 3/4: VAE encode"
launch_per_gpu scripts.create_train_data.step3_vae_encode

# ============================================================================
# Step 4: Pack into sharded LMDB + cache keys
# ============================================================================
echo ""
echo ">>> Step 4/4: Build LMDB"
python -m scripts.create_train_data.step4a_pack_lmdb
python -m scripts.create_train_data.step4b_cache_keys

echo ""
echo "All steps completed on $HOSTNAME."
