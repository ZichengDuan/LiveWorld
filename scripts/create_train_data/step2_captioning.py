#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import torch
from tqdm.auto import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from scripts.create_train_data._utils import get_rank_info, shard_items


_STYLE_TEMPLATES = [
    ("Write a concise, literal caption.", 50, "1-2"),
    ("Write a detailed, cinematic caption.", 120, "2-4"),
    ("Write a calm, observational caption.", 80, "2-3"),
    ("Write a thorough, descriptive caption.", 150, "3-5"),
    ("Write a compact, vivid caption.", 60, "1-2"),
    ("Write a rich, immersive caption.", 100, "2-3"),
]

_FOREGROUND_MAX_WORDS = 35
_FOREGROUND_NUM_SENTENCES = "1"
_FOREGROUND_STYLE_TEMPLATES = [
    "Write a short, literal foreground caption.",
    "Write a compact, action-focused foreground caption.",
    "Write a brief, vivid foreground caption.",
]

_BACKGROUND_MAX_WORDS = 60
_BACKGROUND_NUM_SENTENCES = "1-2"
_BACKGROUND_STYLE_TEMPLATES = [
    "Write a short, concrete scene caption.",
    "Write a compact, atmospheric scene caption.",
    "Write a concise, observational scene caption.",
]


def _select_style(video_id: str, video_key: str) -> tuple[str, int, str]:
    """Returns (style_prompt, max_words, num_sentences)"""
    digest = hashlib.md5(f"{video_id}:{video_key}".encode("utf-8")).hexdigest()
    idx = int(digest, 16) % len(_STYLE_TEMPLATES)
    return _STYLE_TEMPLATES[idx]


def _select_variant(video_id: str, video_key: str, variants: list[str]) -> str:
    digest = hashlib.md5(f"{video_id}:{video_key}:variant".encode("utf-8")).hexdigest()
    idx = int(digest, 16) % len(variants)
    return variants[idx]


def _normalize_caption(text: str, max_words: int) -> str:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return cleaned
    words = cleaned.split()
    if len(words) > max_words:
        cleaned = " ".join(words[:max_words])
    return cleaned


def _parse_video_keys(raw: str) -> list[str]:
    if not raw:
        return []
    keys = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        if item.lower().endswith(".mp4"):
            item = item[:-4]
        keys.append(item)
    return keys


def _infer_caption_mode(video_key: str) -> str:
    key = video_key.lower()
    if "fg_rgb" in key:
        return "foreground"
    if "scene_rgb" in key:
        return "background"
    return "default"


def _resolve_video_path(video_dir: Path, video_key: str, mode: str) -> Path:
    video_path = video_dir / f"{video_key}.mp4"
    if video_path.exists():
        return video_path
    if mode in ("foreground", "background"):
        fallback = video_dir / "train_target_rgb.mp4"
        if fallback.exists():
            return fallback
    return video_path


def _find_sample_meta(video_dir: Path) -> dict | None:
    candidates = [video_dir / "train_sample.json", video_dir / "sample.json"]
    for path in candidates:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return None
    sample_jsons = sorted(video_dir.glob("*sample.json"))
    if sample_jsons:
        try:
            return json.loads(sample_jsons[0].read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _is_mask_all_black(mask_mp4: Path) -> bool:
    if not mask_mp4.exists():
        return True
    cap = cv2.VideoCapture(str(mask_mp4))
    if not cap.isOpened():
        return True
    all_black = True
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame.ndim == 3:
            frame = frame[..., 0]
        if frame.max() > 2:
            all_black = False
            break
    cap.release()
    return all_black


def _detect_has_fg(video_dir: Path) -> bool:
    meta = _find_sample_meta(video_dir)
    if meta and "has_fg" in meta:
        return bool(meta["has_fg"])

    mask_files = sorted(video_dir.glob("mask_*.mp4"))
    if mask_files:
        for mask_path in mask_files:
            name = mask_path.stem.replace("mask_", "").lower()
            if name == "sky":
                continue
            if not _is_mask_all_black(mask_path):
                return True
        return False

    dynamic_mask = video_dir / "dynamic_masks.mp4"
    if dynamic_mask.exists():
        return not _is_mask_all_black(dynamic_mask)

    return False


def _list_video_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        raise NotADirectoryError(root)
    return sorted([p for p in root.iterdir() if p.is_dir()])


def _log_message(
    message: str,
    rank: int,
    world_size: int,
    level: str = "INFO",
    quiet_nonzero: bool = False,
    stage: str | None = None,
) -> None:
    if quiet_nonzero and rank != 0 and level != "ERROR":
        return
    tag = f"[Caption][{level}][R{rank}/{world_size}]"
    if stage:
        tag += f"[{stage}]"
    print(f"{tag} {message}", flush=True)


class Qwen3VLCaptioner:
    def __init__(
        self,
        model_path: str,
        device: str,
        max_new_tokens: int,
        attn_implementation: str,
        caption_prompt: str | None = None,
    ):
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
            device_map={"": device},
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.max_new_tokens = max_new_tokens
        self.caption_prompt = caption_prompt

    def caption(self, video_path: str, video_id: str, video_key: str, mode: str) -> str:
        if self.caption_prompt:
            prompt = self.caption_prompt
            max_words = 150  # default for custom prompt
        elif mode == "foreground":
            style = _select_variant(video_id, video_key, _FOREGROUND_STYLE_TEMPLATES)
            prompt = (
                f"{style} Describe ONLY the foreground dynamic objects "
                "(people, vehicles, animals, handheld objects). "
                "Describe count, appearance, actions, and motion. "
                "Do NOT mention background, scenery, buildings, or camera motion. "
                "If no foreground objects are visible, output: Nothing. "
                f"Use at most {_FOREGROUND_MAX_WORDS} words. Output {_FOREGROUND_NUM_SENTENCES} sentences."
            )
            max_words = _FOREGROUND_MAX_WORDS
        elif mode == "background":
            style = _select_variant(video_id, video_key, _BACKGROUND_STYLE_TEMPLATES)
            prompt = (
                f"{style} Describe ONLY the static background environment. "
                "Ignore foreground objects such as people, vehicles, animals, and handheld items. "
                "Describe setting, surfaces, architecture, lighting, atmosphere, and camera motion. "
                "Do NOT describe any foreground objects. "
                f"Use at most {_BACKGROUND_MAX_WORDS} words. Output {_BACKGROUND_NUM_SENTENCES} sentences."
            )
            max_words = _BACKGROUND_MAX_WORDS
        else:
            style, max_words, num_sentences = _select_style(video_id, video_key)
            prompt = (
                f"{style} Describe what is visible in the scene (objects, environment, people, lighting, atmosphere) "
                "and how the camera moves (e.g., panning left, moving forward, tilting up, rotating, zooming in/out, tracking). "
                "Be specific about the scene details and camera motion direction. "
                "Do NOT mention the camera operator, videographer, photographer, or any person holding/operating the camera. "
                "Use 'person' for any humans visible in the scene. "
                f"Use at most {max_words} words. Output {num_sentences} sentences."
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return _normalize_caption(output_text, max_words=max_words)


def main() -> None:
    from omegaconf import OmegaConf
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    cfg = OmegaConf.load(str(_PROJECT_ROOT / "configs" / "data_preparation.yaml"))

    input_root = cfg.get("caption_input_root", None) or cfg.output_root
    video_keys = _parse_video_keys(cfg.caption_video_keys)
    if not video_keys:
        raise ValueError("caption_video_keys is empty in config.")

    rank, world_size, local_rank = get_rank_info()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    quiet = cfg.get("caption_quiet_nonzero", True)
    _log_message("Loading Qwen3VL model...", rank, world_size, stage="Init", level="INFO", quiet_nonzero=quiet)
    captioner = Qwen3VLCaptioner(
        model_path=cfg.get("caption_model_path", "ckpts/Qwen--Qwen3-VL-8B-Instruct"),
        device=device,
        max_new_tokens=cfg.get("caption_max_new_tokens", 128),
        attn_implementation=cfg.get("caption_attn_implementation", "flash_attention_2"),
        caption_prompt=cfg.get("caption_prompt", None),
    )

    root = Path(input_root)
    video_dirs = _list_video_dirs(root)
    max_videos = cfg.get("caption_max_videos", None)
    if max_videos is not None:
        video_dirs = video_dirs[:max_videos]
    video_dirs = shard_items(video_dirs, rank, world_size)
    skip_existing = cfg.get("caption_skip_existing", True)

    pbar = tqdm(total=len(video_dirs), disable=rank != 0, desc="captioning")
    for video_dir in video_dirs:
        video_id = video_dir.name
        if rank == 0:
            pbar.set_description(f"captioning {video_id}")

        has_fg = _detect_has_fg(video_dir)
        for key in video_keys:
            mode = _infer_caption_mode(key)
            if mode == "foreground" and not has_fg:
                _log_message(
                    f"Skip {video_dir / f'{key}.mp4'} (no foreground detected)",
                    rank,
                    world_size,
                    level="SKIP",
                    quiet_nonzero=quiet,
                )
                continue
            video_path = _resolve_video_path(video_dir, key, mode)
            if not video_path.exists():
                _log_message(f"Missing {video_path}", rank, world_size, level="WARN", quiet_nonzero=quiet)
                continue
            if video_path.name != f"{key}.mp4" and not quiet:
                _log_message(
                    f"Fallback to {video_path.name} for {key}",
                    rank,
                    world_size,
                    level="INFO",
                    quiet_nonzero=quiet,
                )
            out_path = video_dir / f"{key}.txt"
            if skip_existing and out_path.exists() and out_path.read_text(encoding="utf-8").strip():
                _log_message(f"Skip {out_path} (exists)", rank, world_size, level="SKIP", quiet_nonzero=quiet)
                continue
            caption = captioner.caption(str(video_path), video_id=video_id, video_key=key, mode=mode)
            out_path.write_text(caption + "\n", encoding="utf-8")
            _log_message(f"Wrote {out_path}", rank, world_size, quiet_nonzero=quiet)

        if rank == 0:
            pbar.update(1)

    if rank == 0:
        pbar.close()

    _log_message("Done", rank, world_size, stage="Done", level="INFO", quiet_nonzero=quiet)


if __name__ == "__main__":
    main()
