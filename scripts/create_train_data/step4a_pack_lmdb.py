#!/usr/bin/env python3
"""
Pack LiveWorld data folders into sharded LMDB.

Packs .pt (latents) and .txt (captions) files from each sample folder
into a sharded LMDB for efficient training data loading.

支持两种 latent 存储格式：
1. 新格式: latents.pt (所有 latent 合并在一个文件中)
2. 旧格式: 单独的 .pt 文件 (train_target_rgb.pt, train_preceding_rgb_9.pt, ...)

Also extracts and packs the first frame of train_target_rgb.mp4 for I2V training.
Optional foreground projection latents (train_*_proj_fg_rgb*.pt) are included when present.

Usage:
    python -m scripts.dataset_preparation.step4a_pack_lmdb
    python -m scripts.dataset_preparation.step4a_pack_lmdb --data-root data/liveworld/frame33_fps16_2000
    python -m scripts.dataset_preparation.step4a_pack_lmdb --num-workers 16  # 多进程加速

Output directory is automatically derived: {data_root}_lmdb
  e.g., data/liveworld/frame33_fps16_2000 -> data/liveworld/frame33_fps16_2000_lmdb
"""
from __future__ import annotations

import io
import lmdb
import os
import pickle
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

GB = 1024 ** 3

# 合并后的 latents 文件名
LATENTS_FILENAME = "latents.pt"

# Latent keys that should be read from latents.pt (or individual .pt files)
LATENT_KEYS = [
    "train_target_rgb",
    "train_target_scene_proj_rgb",
    "train_target_proj_fg_rgb",
    "train_preceding_rgb_9",
    "train_preceding_scene_proj_rgb_9",
    "train_preceding_proj_fg_rgb_9",
    "train_preceding_rgb_1",
    "train_preceding_scene_proj_rgb_1",
    "train_preceding_proj_fg_rgb_1",
    "train_preceding_rgb",  # Legacy
    "train_preceding_scene_proj_rgb",  # Legacy
    "train_reference_rgb",
    "train_reference_scene_rgb",
    "train_reference_scene_rgb_orig",
    "train_target_scene_proj_rgb_orig",
    "train_reference_instance_00",
    "train_reference_instance_01",
    "train_reference_instance_02",
    "train_reference_instance_03",
    "train_reference_instance_04",
    "train_preceding_scene_proj_fg_overlay_rgb_9",
    "train_preceding_scene_proj_fg_overlay_rgb_1",
    "train_target_scene_proj_fg_overlay_rgb",
]

# Non-latent files to pack directly (txt, json)
NON_LATENT_FILES = [
    ("train_target_rgb", ".txt"),
    ("train_target_scene_rgb", ".txt"),
    ("train_target_fg_rgb", ".txt"),
    ("train_sample", ".json"),
    ("clip", ".txt"),
]


def get_lmdb_root_from_data_root(data_root: str) -> str:
    """
    Derive LMDB output path from data root.
    e.g., data/liveworld/frame33_fps16_2000 -> data/liveworld/frame33_fps16_2000_lmdb
    """
    data_root = data_root.rstrip("/")
    return f"{data_root}_lmdb"


def list_sample_folders(data_root: Path) -> list[Path]:
    """List all sample folders (8-digit numbered folders)."""
    folders = []
    for p in sorted(data_root.iterdir()):
        if p.is_dir() and p.name.isdigit() and len(p.name) == 8:
            folders.append(p)
    return folders


def extract_first_frame(video_path: Path) -> Optional[bytes]:
    """
    Extract the first frame from a video file and return as PNG bytes.
    Returns None if extraction fails.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return None

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Encode as PNG bytes
    img = Image.fromarray(frame_rgb)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def load_latents_from_folder(folder: Path, latent_keys: list[str]) -> dict[str, torch.Tensor]:
    """
    从文件夹加载 latent tensors。
    优先从 latents.pt 读取，如果不存在则尝试读取单独的 .pt 文件。

    Returns:
        dict of {key: tensor}
    """
    latents = {}
    latents_path = folder / LATENTS_FILENAME

    # 优先从合并的 latents.pt 读取
    if latents_path.exists():
        try:
            merged = torch.load(latents_path, map_location="cpu", weights_only=True)
            for key in latent_keys:
                if key in merged:
                    latents[key] = merged[key]
        except Exception:
            pass

    # 对于 latents.pt 中没有的 key，尝试读取单独的 .pt 文件
    for key in latent_keys:
        if key in latents:
            continue  # 已从 latents.pt 读取
        pt_path = folder / f"{key}.pt"
        if pt_path.exists():
            try:
                data = torch.load(pt_path, map_location="cpu", weights_only=True)
                if isinstance(data, dict) and "latent" in data:
                    latents[key] = data["latent"]
                elif isinstance(data, torch.Tensor):
                    latents[key] = data
            except Exception:
                pass

    return latents


def read_sample_data(
    folder: Path,
    latent_keys: list[str],
    non_latent_files: list[tuple[str, str]],
    required_latent_keys: list[str],
    extract_first_frames: list[str] | None = None,
) -> Optional[dict[str, bytes]]:
    """
    Read all files for one sample folder.
    Returns dict of {filename: bytes} or None if required files are missing.

    Args:
        folder: Sample folder path
        latent_keys: List of latent keys to read from latents.pt or individual .pt files
        non_latent_files: List of (stem, extension) for non-latent files (txt, json)
        required_latent_keys: Latent keys that must exist
        extract_first_frames: List of video stems to extract first frame from
                              (e.g., ["train_target_rgb"] -> extracts first frame
                               and saves as "train_target_rgb_frame0.png")
    """
    data = {}

    # Load latents
    latents = load_latents_from_folder(folder, latent_keys)

    # Check required latent keys
    for key in required_latent_keys:
        if key not in latents:
            return None

    # Serialize latents to bytes (as individual .pt format for compatibility)
    for key, tensor in latents.items():
        # 保存为 {"latent": tensor} 格式，与旧格式兼容
        buffer = io.BytesIO()
        torch.save({"latent": tensor}, buffer)
        data[f"{key}.pt"] = buffer.getvalue()

    # Read non-latent files
    for stem, ext in non_latent_files:
        file_path = folder / f"{stem}{ext}"
        filename = f"{stem}{ext}"

        if file_path.exists():
            with open(file_path, "rb") as f:
                data[filename] = f.read()

    # Check required non-latent files
    required_non_latent = [("train_sample", ".json")]
    for stem, ext in required_non_latent:
        if f"{stem}{ext}" not in data:
            return None

    # Extract first frames from specified videos
    if extract_first_frames:
        for video_stem in extract_first_frames:
            video_path = folder / f"{video_stem}.mp4"
            if video_path.exists():
                frame_bytes = extract_first_frame(video_path)
                if frame_bytes is not None:
                    # Save as {video_stem}_frame0.png
                    data[f"{video_stem}_frame0.png"] = frame_bytes

    return data


def _process_single_folder(args: tuple) -> tuple[str, bytes | None]:
    """
    Worker function to process a single folder.
    Returns (folder_name, pickled_data) or (folder_name, None) if invalid.
    """
    folder_str, latent_keys, non_latent_files, required_latent_keys, extract_first_frames = args
    folder = Path(folder_str)
    data = read_sample_data(folder, latent_keys, non_latent_files, required_latent_keys, extract_first_frames)
    if data is None:
        return (folder.name, None)
    return (folder.name, pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))


def estimate_sample_size(
    folders: list[Path],
    latent_keys: list[str],
    non_latent_files: list[tuple[str, str]],
    num_samples: int = 10,
) -> int:
    """Estimate average sample size by checking a few samples."""
    sample_folders = random.sample(folders, min(num_samples, len(folders)))
    total_size = 0
    count = 0

    for folder in sample_folders:
        # Check latents.pt first
        latents_path = folder / LATENTS_FILENAME
        if latents_path.exists():
            total_size += latents_path.stat().st_size
        else:
            # Fallback to individual .pt files
            for key in latent_keys:
                pt_path = folder / f"{key}.pt"
                if pt_path.exists():
                    total_size += pt_path.stat().st_size

        # Non-latent files
        for stem, ext in non_latent_files:
            file_path = folder / f"{stem}{ext}"
            if file_path.exists():
                total_size += file_path.stat().st_size
        count += 1

    return int(total_size / count) if count > 0 else 0


def safe_put_batch(
    env: lmdb.Environment,
    buffer: list[tuple[bytes, bytes]],
    grow_factor: float = 1.5,
) -> None:
    """Write a batch to LMDB, growing map size if needed."""
    while True:
        try:
            with env.begin(write=True) as txn:
                for k, v in buffer:
                    txn.put(k, v, overwrite=True)
            return
        except lmdb.MapFullError:
            current = env.info().get("map_size", 0)
            new_size = int(current * grow_factor) if current else int(1.5 * GB)
            env.set_mapsize(new_size)
            print(f"  LMDB map full; expanded to {new_size / GB:.2f} GB")


def build_sharded_lmdb(
    data_root: str,
    lmdb_root: str,
    latent_keys: list[str],
    non_latent_files: list[tuple[str, str]],
    required_latent_keys: list[str],
    extract_first_frames: list[str] | None = None,
    target_shard_size_gb: float = 10.0,
    write_batch: int = 256,
    num_workers: int = 0,
):
    """
    Build sharded LMDB from LiveWorld data folders.

    Args:
        data_root: Root directory containing sample folders (00000000, 00000001, ...)
        lmdb_root: Output LMDB directory
        latent_keys: List of latent keys to read
        non_latent_files: List of (stem, extension) for non-latent files
        required_latent_keys: Latent keys that must exist
        extract_first_frames: List of video stems to extract first frame from
        target_shard_size_gb: Target size per shard in GB
        write_batch: Samples per transaction
        num_workers: Number of parallel workers (0 = auto, based on CPU count)
    """
    data_root = Path(data_root).resolve()
    lmdb_root = Path(lmdb_root).resolve()

    print(f"Data root: {data_root}")
    print(f"LMDB root: {lmdb_root}")

    # List sample folders
    print("\nListing sample folders...")
    folders = list_sample_folders(data_root)
    print(f"Found {len(folders)} sample folders")

    if not folders:
        raise RuntimeError(f"No sample folders found in {data_root}")

    # Estimate sample size from a few samples (快速估算)
    print("\nEstimating sample sizes...")
    avg_size = estimate_sample_size(folders[:100], latent_keys, non_latent_files)
    print(f"Average sample size: {avg_size / 1024**2:.2f} MB")

    # Calculate shard size
    target_bytes = target_shard_size_gb * GB
    shard_size = max(1, int(target_bytes / avg_size)) if avg_size > 0 else 1000
    num_shards = (len(folders) + shard_size - 1) // shard_size

    # Determine number of workers
    if num_workers <= 0:
        num_workers = min(os.cpu_count() or 4, 32)

    print(f"\nShard configuration:")
    print(f"  Target shard size: {target_shard_size_gb} GB")
    print(f"  Samples per shard: {shard_size}")
    print(f"  Total shards (max): {num_shards}")
    print(f"  Parallel workers: {num_workers}")

    # Create output directory
    lmdb_root.mkdir(parents=True, exist_ok=True)

    # Build shards - 直接打包，跳过无效样本
    folder_idx = 0
    total_valid = 0
    total_skipped = 0

    for shard_id in range(num_shards):
        shard_folders = folders[folder_idx:folder_idx + shard_size]
        folder_idx += shard_size

        if not shard_folders:
            break

        shard_dir = lmdb_root / f"shard_{shard_id:03d}.lmdb"
        shard_dir.mkdir(parents=True, exist_ok=True)

        # Estimate map size for this shard
        est_bytes = avg_size * len(shard_folders)
        map_size = max(int(est_bytes * 3), int(est_bytes + 2 * GB), 1 << 30)

        print(f"\n{'='*60}")
        print(f"Building shard {shard_id}/{num_shards-1}: {shard_dir.name}")
        print(f"  Folders to process: {len(shard_folders)}")
        print(f"  Map size: {map_size / GB:.2f} GB")

        env = lmdb.open(
            str(shard_dir),
            map_size=map_size,
            subdir=True,
            readonly=False,
            lock=True,
            readahead=False,
            meminit=False,
            sync=True,
            max_readers=256,
        )

        buffer = []
        shard_valid = 0
        shard_skipped = 0

        # 多进程并行读取
        if num_workers > 1:
            # Prepare args for workers
            worker_args = [
                (str(f), latent_keys, non_latent_files, required_latent_keys, extract_first_frames)
                for f in shard_folders
            ]

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(_process_single_folder, args): args[0] for args in worker_args}

                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Shard {shard_id}"):
                    folder_name, pickled_data = future.result()
                    if pickled_data is None:
                        shard_skipped += 1
                        continue

                    key = folder_name.encode("utf-8")
                    buffer.append((key, pickled_data))
                    shard_valid += 1

                    if len(buffer) >= write_batch:
                        safe_put_batch(env, buffer)
                        buffer.clear()
        else:
            # 单进程模式
            for folder in tqdm(shard_folders, desc=f"Shard {shard_id}"):
                data = read_sample_data(folder, latent_keys, non_latent_files, required_latent_keys, extract_first_frames)
                if data is None:
                    shard_skipped += 1
                    continue

                key = folder.name.encode("utf-8")
                value = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
                buffer.append((key, value))
                shard_valid += 1

                if len(buffer) >= write_batch:
                    safe_put_batch(env, buffer)
                    buffer.clear()

        if buffer:
            safe_put_batch(env, buffer)
            buffer.clear()

        env.sync()
        env.close()

        total_valid += shard_valid
        total_skipped += shard_skipped
        print(f"  Finished shard {shard_id}: {shard_valid} valid, {shard_skipped} skipped")

    print(f"\n{'='*60}")
    print(f"All shards complete!")
    print(f"  Total valid: {total_valid}")
    print(f"  Total skipped: {total_skipped}")
    print(f"Output: {lmdb_root}")
    print(f"{'='*60}")


def main():
    from omegaconf import OmegaConf
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    cfg = OmegaConf.load(str(_PROJECT_ROOT / "configs" / "data_preparation.yaml"))

    data_root = cfg.get("lmdb_data_root", None) or cfg.output_root
    lmdb_root = get_lmdb_root_from_data_root(data_root)
    target_shard_size_gb = cfg.get("lmdb_target_shard_size_gb", 30.0)
    num_workers = cfg.get("lmdb_num_workers", 0)

    # Latent keys to read (from latents.pt or individual .pt files)
    latent_keys = LATENT_KEYS

    # Non-latent files to pack
    non_latent_files = NON_LATENT_FILES

    # Required latent keys (sample is skipped if these are missing)
    required_latent_keys = ["train_target_rgb"]

    # Videos to extract first frame from (for I2V training)
    extract_first_frames = ["train_target_rgb"]

    print("="*60)
    print("LiveWorld LMDB Packer")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Data root: {data_root}")
    print(f"  LMDB root: {lmdb_root}")
    print(f"  Target shard size: {target_shard_size_gb} GB")
    print(f"  Latent keys: {latent_keys}")
    print(f"  Non-latent files: {[f'{s}{e}' for s, e in non_latent_files]}")
    print(f"  Required latent keys: {required_latent_keys}")
    print(f"  Extract first frames: {[f'{s}.mp4 -> {s}_frame0.png' for s in extract_first_frames]}")
    print(f"  Supports: latents.pt (merged) or individual .pt files")
    print(f"  Parallel workers: {num_workers} (0 = auto)")
    print("-"*60)

    build_sharded_lmdb(
        data_root=data_root,
        lmdb_root=lmdb_root,
        latent_keys=latent_keys,
        non_latent_files=non_latent_files,
        required_latent_keys=required_latent_keys,
        extract_first_frames=extract_first_frames,
        target_shard_size_gb=target_shard_size_gb,
        write_batch=256,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    main()
