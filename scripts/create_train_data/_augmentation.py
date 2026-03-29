#!/usr/bin/env python3
"""
Post-processing: overlay foreground projection onto scene projection in pixel space.

Reads existing scene projection and fg projection mp4 files, overlays fg non-black
pixels onto scene, and saves new overlay mp4 files.

With --augment: randomly degrade the fg region before overlaying to simulate
imperfect projection (point cloud holes, occlusion, projection artifacts).

Multi-node: folders sharded by NODE_RANK/NUM_NODES env vars.
Multi-thread: each node uses --num-workers threads (default: CPU count).

Usage:
    # Single machine (no augmentation)
    python -m scripts.dataset_preparation._augmentation

    # With augmentation
    python -m scripts.dataset_preparation._augmentation --augment --no-skip-existing

    # Multi-node (set env vars per node, or use augmentation)
    NODE_RANK=0 NUM_NODES=4 python -m scripts.dataset_preparation._augmentation --augment
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from omegaconf import OmegaConf
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG = OmegaConf.load(str(_PROJECT_ROOT / "configs" / "data_preparation.yaml"))
from tqdm import tqdm


# ---------------------------------------------------------------------------
# FG overlay augmentation
# ---------------------------------------------------------------------------

def augment_fg_overlay(
    fg_frames: np.ndarray,
    blob_drop_prob: float = 0.7,
    blob_count_min: int = 3,
    blob_count_max: int = 15,
    blob_radius_min: int = 5,
    blob_radius_max: int = 30,
    line_prob: float = 0.5,
    line_count_min: int = 2,
    line_count_max: int = 8,
    line_width_min: int = 1,
    line_width_max: int = 5,
    block_drop_prob: float = 0.3,
    block_drop_ratio_min: float = 0.1,
    block_drop_ratio_max: float = 0.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Augment fg frames by degrading non-black pixel regions.

    Simulates imperfect projection artifacts: point cloud holes (blob drop),
    projection splits (lines), and occlusion (block drop).

    All operations only affect pixels where fg > 10 (the non-black fg region).
    Parameters are sampled once per video and applied consistently across frames.

    Args:
        fg_frames: [T, H, W, 3] uint8 foreground frames (black background).
        blob_drop_prob: Probability of enabling blob drop augmentation.
        blob_count_min/max: Range for number of circular blobs to drop.
        blob_radius_min/max: Range for blob radius in pixels.
        line_prob: Probability of enabling line augmentation.
        line_count_min/max: Range for number of lines to draw.
        line_width_min/max: Range for line width in pixels.
        block_drop_prob: Probability of enabling block drop augmentation.
        block_drop_ratio_min/max: Range for block area as fraction of fg bbox.
        rng: Random number generator for reproducibility.

    Returns:
        Augmented fg frames [T, H, W, 3] uint8.
    """
    if rng is None:
        rng = np.random.default_rng()

    T, H, W, _ = fg_frames.shape

    # Compute fg mask across all frames: union of non-black pixels.
    fg_mask = (fg_frames > 10).any(axis=-1)  # [T, H, W]
    union_mask = fg_mask.any(axis=0)  # [H, W]

    if not union_mask.any():
        return fg_frames

    # Bounding box of fg region (union across frames).
    ys, xs = np.where(union_mask)
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    bbox_h = y_max - y_min + 1
    bbox_w = x_max - x_min + 1

    if bbox_h < 4 or bbox_w < 4:
        return fg_frames

    out = fg_frames.copy()

    # Build a degradation mask: True = keep, False = set to black.
    # Start with all True, augmentations punch holes.
    keep_mask = np.ones((H, W), dtype=np.uint8)

    # --- 1. Blob drop: circular patches within fg bbox ---
    if rng.random() < blob_drop_prob:
        n_blobs = rng.integers(blob_count_min, blob_count_max + 1)
        for _ in range(n_blobs):
            radius = rng.integers(blob_radius_min, blob_radius_max + 1)
            cy = rng.integers(y_min, y_max + 1)
            cx = rng.integers(x_min, x_max + 1)
            cv2.circle(keep_mask, (int(cx), int(cy)), int(radius), 0, -1)

    # --- 2. Lines: random lines through fg bbox ---
    if rng.random() < line_prob:
        n_lines = rng.integers(line_count_min, line_count_max + 1)
        for _ in range(n_lines):
            width = rng.integers(line_width_min, line_width_max + 1)
            # Random start/end within the fg bounding box.
            x1 = rng.integers(x_min, x_max + 1)
            y1 = rng.integers(y_min, y_max + 1)
            x2 = rng.integers(x_min, x_max + 1)
            y2 = rng.integers(y_min, y_max + 1)
            cv2.line(keep_mask, (int(x1), int(y1)), (int(x2), int(y2)), 0, int(width))

    # --- 3. Block drop: large rectangle within fg bbox ---
    if rng.random() < block_drop_prob:
        ratio = rng.uniform(block_drop_ratio_min, block_drop_ratio_max)
        block_area = bbox_h * bbox_w * ratio
        # Random aspect ratio between 0.5 and 2.0.
        aspect = rng.uniform(0.5, 2.0)
        bh = int(np.sqrt(block_area / aspect))
        bw = int(np.sqrt(block_area * aspect))
        bh = min(bh, bbox_h)
        bw = min(bw, bbox_w)
        if bh > 0 and bw > 0:
            by = rng.integers(y_min, max(y_min + 1, y_max - bh + 2))
            bx = rng.integers(x_min, max(x_min + 1, x_max - bw + 2))
            keep_mask[by:by + bh, bx:bx + bw] = 0

    # Apply degradation: zero out fg pixels where keep_mask is False.
    # Only affects pixels that are in the fg mask (non-black).
    drop_mask_2d = keep_mask == 0  # [H, W], True = drop
    for t in range(T):
        frame_drop = fg_mask[t] & drop_mask_2d  # only drop within this frame's fg
        out[t][frame_drop] = 0

    return out


def overlay_fg_on_scene(scene_frames: np.ndarray, fg_frames: np.ndarray) -> np.ndarray:
    """Overlay fg non-black pixels onto scene frames.

    Args:
        scene_frames: [T, H, W, 3] uint8
        fg_frames: [T, H, W, 3] uint8

    Returns:
        [T, H, W, 3] uint8 with fg pixels overlaid on scene
    """
    fg_mask = (fg_frames > 10).any(axis=-1, keepdims=True)
    return np.where(fg_mask, fg_frames, scene_frames)


def load_video(path: Path) -> np.ndarray | None:
    """Load video as [T, H, W, 3] uint8 array."""
    if not path.exists():
        return None
    reader = imageio.get_reader(str(path))
    frames = []
    for frame in reader:
        frames.append(frame)
    reader.close()
    return np.stack(frames, axis=0)


def save_video(frames: np.ndarray, path: Path, fps: float = 16.0) -> None:
    """Save [T, H, W, 3] uint8 array as mp4."""
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), frames, fps=fps, macro_block_size=None)


def process_folder(
    folder: Path,
    skip_existing: bool = True,
    augment: bool = False,
    aug_kwargs: dict | None = None,
) -> dict:
    """Process one sample folder, generating overlay videos.

    Args:
        folder: Sample folder path.
        skip_existing: Skip if overlay already exists.
        augment: Apply fg augmentation before overlay.
        aug_kwargs: Keyword arguments for augment_fg_overlay().

    Returns dict with counts of generated/skipped/missing.
    """
    pairs = [
        ("train_target_scene_proj_rgb.mp4", "train_target_proj_fg_rgb.mp4", "train_target_scene_proj_fg_overlay_rgb.mp4"),
        ("train_preceding_scene_proj_rgb_9.mp4", "train_preceding_proj_fg_rgb_9.mp4", "train_preceding_scene_proj_fg_overlay_rgb_9.mp4"),
        ("train_preceding_scene_proj_rgb_1.mp4", "train_preceding_proj_fg_rgb_1.mp4", "train_preceding_scene_proj_fg_overlay_rgb_1.mp4"),
    ]

    result = {"generated": 0, "skipped": 0, "missing": 0}

    for scene_name, fg_name, overlay_name in pairs:
        overlay_path = folder / overlay_name

        if skip_existing and overlay_path.exists():
            result["skipped"] += 1
            continue

        scene_path = folder / scene_name
        fg_path = folder / fg_name

        if not scene_path.exists() or not fg_path.exists():
            result["missing"] += 1
            continue

        scene_frames = load_video(scene_path)
        fg_frames = load_video(fg_path)

        if scene_frames is None or fg_frames is None:
            result["missing"] += 1
            continue

        # Handle frame count mismatch (truncate to shorter)
        min_t = min(len(scene_frames), len(fg_frames))
        scene_frames = scene_frames[:min_t]
        fg_frames = fg_frames[:min_t]

        if augment:
            fg_frames = augment_fg_overlay(fg_frames, **(aug_kwargs or {}))

        overlay = overlay_fg_on_scene(scene_frames, fg_frames)
        save_video(overlay, overlay_path)
        result["generated"] += 1

    return result


def main():
    parser = argparse.ArgumentParser(description="Overlay fg projection on scene projection")
    parser.add_argument("--input-root", type=str, default=None,
                        help="Root directory containing sample folders (default: from config)")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Number of threads per node (0 = CPU count)")

    # Augmentation
    parser.add_argument("--augment", action="store_true", default=False,
                        help="Randomly degrade fg before overlay (simulate imperfect projection)")
    parser.add_argument("--blob-drop-prob", type=float, default=0.7)
    parser.add_argument("--blob-count-min", type=int, default=3)
    parser.add_argument("--blob-count-max", type=int, default=15)
    parser.add_argument("--blob-radius-min", type=int, default=5)
    parser.add_argument("--blob-radius-max", type=int, default=30)
    parser.add_argument("--line-prob", type=float, default=0.5)
    parser.add_argument("--line-count-min", type=int, default=2)
    parser.add_argument("--line-count-max", type=int, default=8)
    parser.add_argument("--line-width-min", type=int, default=1)
    parser.add_argument("--line-width-max", type=int, default=5)
    parser.add_argument("--block-drop-prob", type=float, default=0.3)
    parser.add_argument("--block-drop-ratio-min", type=float, default=0.1)
    parser.add_argument("--block-drop-ratio-max", type=float, default=0.5)

    args = parser.parse_args()

    if args.input_root is None:
        input_root = CONFIG.output_root
    else:
        input_root = args.input_root

    input_root = Path(input_root).resolve()

    # Multi-node sharding
    node_rank = int(os.environ.get("NODE_RANK", 0))
    num_nodes = int(os.environ.get("NUM_NODES", 1))

    # Thread count
    num_workers = args.num_workers if args.num_workers > 0 else (os.cpu_count() or 4)

    # Augmentation kwargs
    aug_kwargs = {
        "blob_drop_prob": args.blob_drop_prob,
        "blob_count_min": args.blob_count_min,
        "blob_count_max": args.blob_count_max,
        "blob_radius_min": args.blob_radius_min,
        "blob_radius_max": args.blob_radius_max,
        "line_prob": args.line_prob,
        "line_count_min": args.line_count_min,
        "line_count_max": args.line_count_max,
        "line_width_min": args.line_width_min,
        "line_width_max": args.line_width_max,
        "block_drop_prob": args.block_drop_prob,
        "block_drop_ratio_min": args.block_drop_ratio_min,
        "block_drop_ratio_max": args.block_drop_ratio_max,
    }

    # List sample folders
    folders = sorted(
        p for p in input_root.iterdir()
        if p.is_dir() and p.name.isdigit() and len(p.name) == 8
    )

    # Shard by node
    my_folders = folders[node_rank::num_nodes]

    aug_label = " (augment ON)" if args.augment else ""
    print(f"[Node {node_rank}/{num_nodes}] Processing {len(my_folders)}/{len(folders)} folders "
          f"with {num_workers} threads from {input_root}{aug_label}")

    total = {"generated": 0, "skipped": 0, "missing": 0}

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                process_folder, folder, args.skip_existing,
                augment=args.augment, aug_kwargs=aug_kwargs,
            ): folder
            for folder in my_folders
        }
        pbar = tqdm(total=len(futures), desc=f"Node {node_rank} overlay", mininterval=1.0)
        for future in as_completed(futures):
            result = future.result()
            for k, v in result.items():
                total[k] += v
            pbar.set_postfix(gen=total["generated"], skip=total["skipped"], miss=total["missing"])
            pbar.update(1)
        pbar.close()

    print(f"[Node {node_rank}] Done: generated={total['generated']}, "
          f"skipped={total['skipped']}, missing={total['missing']}")


if __name__ == "__main__":
    main()
