#!/usr/bin/env python3
"""
Cache LMDB keys for fast dataset initialization.

Reads all keys from sharded LMDB and saves them as pickle files
for fast loading during dataset initialization.

Usage:
    python -m scripts.dataset_preparation.step4b_cache_keys
    python -m scripts.dataset_preparation.step4b_cache_keys --data-root data/liveworld/frame33_fps16_2000

LMDB directory is automatically derived: {data_root}_lmdb
"""
from __future__ import annotations

import lmdb
import pickle
from pathlib import Path
from tqdm import tqdm


def get_lmdb_root_from_data_root(data_root: str) -> str:
    """
    Derive LMDB path from data root.
    e.g., data/liveworld/frame33_fps16_2000 -> data/liveworld/frame33_fps16_2000_lmdb
    """
    data_root = data_root.rstrip("/")
    return f"{data_root}_lmdb"


def cache_lmdb_keys(lmdb_dir: str, force_rebuild: bool = False) -> None:
    """
    Cache all keys from LMDB to disk.

    Args:
        lmdb_dir: Directory containing LMDB shards
        force_rebuild: If True, rebuild cache even if it exists
    """
    lmdb_path = Path(lmdb_dir)

    if not lmdb_path.exists():
        raise FileNotFoundError(f"LMDB directory not found: {lmdb_path}")

    # Detect if sharded or single LMDB
    is_sharded = not (lmdb_path / "data.mdb").exists()

    if is_sharded:
        print("Detected sharded LMDB format")
        cache_sharded_keys(lmdb_path, force_rebuild)
    else:
        print("Detected single LMDB format")
        cache_single_keys(lmdb_path, force_rebuild)


def cache_single_keys(lmdb_path: Path, force_rebuild: bool = False) -> None:
    """Cache keys for single LMDB."""
    cache_file = lmdb_path / "keys_cache.pkl"

    if cache_file.exists() and not force_rebuild:
        print(f"Keys cache already exists: {cache_file}")
        print("Use --force to regenerate")
        return

    print(f"Caching keys from: {lmdb_path}")

    env = lmdb.open(
        str(lmdb_path),
        readonly=True,
        lock=False,
        readahead=True,
        meminit=False,
        max_readers=2048,
    )

    with env.begin(write=False) as txn:
        print("Reading all keys...")
        keys = [key for key, _ in tqdm(txn.cursor(), desc="Scanning keys")]

    env.close()

    print(f"Saving {len(keys)} keys to cache...")
    with open(cache_file, "wb") as f:
        pickle.dump(keys, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"  Keys cache saved: {cache_file}")
    print(f"  Total keys: {len(keys)}")


def cache_sharded_keys(lmdb_root: Path, force_rebuild: bool = False) -> None:
    """Cache keys for sharded LMDB."""
    cache_file = lmdb_root / "sharded_keys_cache.pkl"

    if cache_file.exists() and not force_rebuild:
        print(f"Keys cache already exists: {cache_file}")
        print("Use --force to regenerate")
        return

    # Find all shards
    shard_dirs = sorted([
        d for d in lmdb_root.iterdir()
        if d.is_dir() and d.name.startswith("shard_") and d.name.endswith(".lmdb")
    ])

    if not shard_dirs:
        raise ValueError(f"No shards found in {lmdb_root}")

    print(f"Found {len(shard_dirs)} shards")

    # Read keys from each shard
    all_shard_keys = []
    shard_offsets = [0]
    total_keys = 0

    for shard_dir in tqdm(shard_dirs, desc="Processing shards"):
        env = lmdb.open(
            str(shard_dir),
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
            max_readers=2048,
        )

        with env.begin(write=False) as txn:
            keys = [key for key, _ in txn.cursor()]

        env.close()

        all_shard_keys.append(keys)
        total_keys += len(keys)
        shard_offsets.append(total_keys)

        print(f"  {shard_dir.name}: {len(keys)} keys")

    # Save cache
    cache_data = {
        "shard_keys": all_shard_keys,
        "shard_offsets": shard_offsets,
        "shard_paths": [str(d) for d in shard_dirs],
        "total_keys": total_keys,
    }

    print(f"\nSaving cache for {total_keys} total keys...")
    with open(cache_file, "wb") as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"  Sharded keys cache saved: {cache_file}")
    print(f"  Total shards: {len(shard_dirs)}")
    print(f"  Total keys: {total_keys}")


def main():
    from omegaconf import OmegaConf
    from pathlib import Path
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    cfg = OmegaConf.load(str(_PROJECT_ROOT / "configs" / "data_preparation.yaml"))

    data_root = cfg.get("lmdb_data_root", None) or cfg.output_root
    lmdb_dir = get_lmdb_root_from_data_root(data_root)
    force_rebuild = cfg.get("lmdb_force_cache", True)

    print("=" * 60)
    print("LMDB Keys Cache Generator")
    print("=" * 60)
    print(f"\nLMDB directory: {lmdb_dir}")
    print("-" * 60)

    try:
        cache_lmdb_keys(lmdb_dir, force_rebuild=force_rebuild)
        print("\n" + "=" * 60)
        print("  Keys cache generation complete!")
        print("=" * 60)
    except Exception as e:
        print(f"\n  Error: {e}")
        raise


if __name__ == "__main__":
    main()
