#!/usr/bin/env python3
"""Assemble per-image event bench folders with storyline + inference configs.

Final version: case1 (pure horizontal translation) + case2 (two-segment + lookback).

For each image produces:
  - source_image copy
  - entities.json (Qwen entity detection)
  - storyline.json (event evolution script)
  - trajectory_templates/<combo_id>/plan.json + geometry.npz
  - infer_scripts/<combo_id>.yaml
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import cv2
import numpy as np
import torch
import yaml

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

if TYPE_CHECKING:
    from liveworld.pipelines.monitor_centric.qwen_extractor import Qwen3VLEntityExtractor

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
_SCENE_CAMERA_EGO_PATTERNS = (
    r"\bcamera\b.{0,28}\b(?:face|faces|facing|turn|turns|turned|turning|point|points|pointing)\b",
    r"\bfaces?\s+(?:forward|backward|left|right)\b",
    r"\bfacing\s+(?:forward|backward|left|right)\b",
    r"\b(?:forward|backward)\b",
    r"\b(?:clockwise|counterclockwise|ccw|cw)\b",
)

_FG_LOCAL_MOTION_PATTERNS = (
    r"\bturn(?:s|ed|ing)?\s+(?:its|their|his|her)\s+head\b",
    r"\b(?:raise|raises|raised|raising|lift|lifts|lifted|lifting)\s+(?:its|their|his|her)\s+head\b",
    r"\b(?:look|looks|looked|looking)\s+(?:around|left|right|up|down)\b",
    r"\b(?:nod|nods|nodding|tilt|tilts|tilted|tilting)\b",
    r"\b(?:wag|wags|wagging)\s+(?:its|their|his|her)\s+tail\b",
    r"\b(?:glance|glances|glanced|glancing|peek|peeks|peeking)\b",
    r"扭头|抬头|低头|转头|点头|摇头|摆尾|眨眼|张望",
)

_FG_TRAVEL_MOTION_PATTERN = (
    r"\b(move|moves|moved|moving|walk|walks|walked|walking|"
    r"travel|travels|traveled|traveling|cross|crosses|crossed|crossing|"
    r"relocate|relocates|relocated|relocating|traverse|traverses|traversed|traversing|"
    r"advance|advances|advanced|advancing|approach|approaches|approached|approaching|"
    r"tro(t|tt)|trots|trotted|trotting|stride|strides|striding|"
    r"head|heads|headed|heading|stroll|strolls|strolled|strolling|"
    r"wander|wanders|wandered|wandering|pad|pads|padded|padding)\b"
)

_FG_STATIONARY_MOTION_PATTERNS = (
    r"\b(?:stay|stays|stayed|staying|remain|remains|remained|remaining|stand|stands|standing|"
    r"linger|lingers|lingered|lingering|idle|idles|idling)\b.{0,36}"
    r"\b(?:same (?:place|spot|area|position)|in place|on the spot|nearby|near its start|near their start)\b",
    r"\b(?:barely move|barely moves|slight shift|small shift|little movement|minor movement|without moving)\b",
    r"原地|不动|几乎不动|小幅|微小|停在原位",
)

_FG_OUT_OF_FRAME_PATTERNS = (
    r"\b(?:out of frame|outside the frame|off[- ]screen|off screen|leave(?:s|d|ing)? (?:the )?(?:frame|view)|"
    r"exit(?:s|ed|ing)? (?:the )?(?:frame|view)|disappear(?:s|ed|ing)? from (?:the )?(?:frame|view))\b",
    r"走出画面|离开画面|出框|出画|离开可见区域|消失在画面",
)

_FG_DISPLACEMENT_CUE_PATTERNS = (
    r"\bfrom\b.{0,48}\bto\b",
    r"\b(?:across|toward|towards|into|through|to another (?:part|area)|to a different (?:part|area)|"
    r"from one side to the other)\b",
    r"\b(?:toward|towards)\s+the\s+\w+",
    r"\b(?:over to|up to|next to|beside|around)\s+the\s+\w+",
    r"\b(?:onto|off of|down from|away from)\s+the\s+\w+",
    r"从.{0,20}到|穿过|移动到另一个|移动到不同|走向|跑向|靠近",
)

_FG_IN_FRAME_CUE_PATTERNS = (
    r"\b(?:inside|within|in)\s+the\s+frame\b",
    r"\b(?:visible area|in view|within view)\b",
    r"画面内|可见区域内|不出画|不离开画面",
)

_FG_REWRITE_MAX_ATTEMPTS = 3

_FG_MOTION_HARD_RULES = """\
CRITICAL MOTION RULES (must follow):
- Prioritize INTERACTION with visible objects in the scene. Entities should move TOWARD
  a specific object or location (a bench, a table, a rock, a tree, a ball, a food bowl,
  a doorway, etc.) and interact with it (approach, sniff, sit on, lean against,
  circle around, settle beside, etc.).
- If no obvious object is nearby, the entity should travel to a clearly different area
  of the scene (from one side to the other, toward a visible landmark, etc.).
- Describe clear displacement AND a purpose/destination, not aimless wandering.
- Explicitly state WHERE the entity moves (e.g., "walks toward the bench", "trots over
  to the flower pot", "moves from the left side to the table on the right").
- ONLY gentle, steady motions: walking, strolling, trotting, padding, wandering.
  FORBIDDEN intense motions: running, dashing, jumping, leaping, hopping, sprinting,
  lunging, pouncing, chasing, bolting, galloping. These cause unstable video.
- FORBIDDEN local-only motions: turning head, raising/lowering head, nodding, glancing,
  looking around, tail wagging, tiny posture shifts, standing still.
- Each entity must end in a visibly different position than where it started.
- Keep all entities fully inside frame at all times. Never leave the visible area.
- Output 1-2 concise sentences only.
"""


# ---------------------------------------------------------------------------
# Qwen loader
# ---------------------------------------------------------------------------

def _load_qwen_extractor_class():
    """Load Qwen extractor without relying on event_centric package __init__."""
    try:
        from liveworld.pipelines.monitor_centric.qwen_extractor import Qwen3VLEntityExtractor
        return Qwen3VLEntityExtractor
    except Exception as first_exc:
        module_path = (
            Path(_PROJECT_ROOT)
            / "liveworld" / "pipelines" / "monitor_centric" / "qwen_extractor.py"
        )
        if not module_path.exists():
            raise ImportError(
                f"Failed to import Qwen3VLEntityExtractor and fallback not found: {module_path}"
            ) from first_exc
        spec = importlib.util.spec_from_file_location(
            "liveworld.pipelines.monitor_centric.qwen_extractor_standalone",
            str(module_path),
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to create import spec for {module_path}") from first_exc
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        extractor_cls = getattr(module, "Qwen3VLEntityExtractor", None)
        if extractor_cls is None:
            raise ImportError(
                "Fallback module loaded but Qwen3VLEntityExtractor is missing"
            ) from first_exc
        print("[qwen] fallback import via file path")
        return extractor_cls


def _load_event_bench_module(module_stem: str):
    """Load module under scripts/create_infer_sample with robust fallback."""
    module_name = f"scripts.create_infer_sample.{module_stem}"
    try:
        return importlib.import_module(module_name)
    except Exception as first_exc:
        module_path = Path(_PROJECT_ROOT) / "scripts" / "create_infer_sample" / f"{module_stem}.py"
        if not module_path.exists():
            raise ImportError(
                f"Failed to import {module_name} and fallback not found: {module_path}"
            ) from first_exc
        spec = importlib.util.spec_from_file_location(
            f"scripts.create_infer_sample.{module_stem}_standalone",
            str(module_path),
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to create import spec for {module_path}") from first_exc
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_image_files(images_dir: Path, glob_pattern: str = "*") -> List[Path]:
    files: List[Path] = []
    for p in sorted(images_dir.glob(glob_pattern)):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files.append(p)
    return files


def _find_plan_dirs(trajectory_root: Path) -> List[Path]:
    """Find all template directories containing plan.json."""
    dirs: List[Path] = []
    for d in sorted(trajectory_root.iterdir()):
        if d.is_dir() and (d / "plan.json").exists():
            dirs.append(d)
    return dirs


def _extract_combo_id(template_dir_name: str) -> str:
    """Strip image stem prefix to get combo_id."""
    combo_with_seed = template_dir_name
    for marker in ("_case",):
        idx = template_dir_name.find(marker)
        if idx > 0:
            combo_with_seed = template_dir_name[idx + 1:]
            break
    seed_match = re.search(r"_seed\d+$", combo_with_seed)
    if seed_match:
        return combo_with_seed[: seed_match.start()]
    return combo_with_seed


def _load_event_prompts(system_config_path: str) -> Tuple[str, str]:
    """Load prompts from system_config.yaml."""
    with open(system_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return (
        cfg["event"]["evolution"]["system_prompt"],
        cfg["event"]["detection"]["scene_detect_prompt"],
    )


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _read_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_scene_meta_path(scene_pc_dir: Path) -> Optional[Path]:
    """Find generated scene meta file under scene_pointcloud directory."""
    if not scene_pc_dir.exists():
        return None
    metas = sorted(scene_pc_dir.glob("*_scene_meta.npz"))
    return metas[0] if metas else None


def _overlay_output_dir(image_dir: Path, image_stem: str) -> Path:
    return image_dir / "trajectory_templates" / "visualizations" / image_stem


def _overlay_png_path(image_dir: Path, image_stem: str) -> Path:
    return _overlay_output_dir(image_dir, image_stem) / "trajectories_global_overlay.png"


def _cleanup_failed_image_output(image_dir: Path, image_stem: str, reason: str) -> None:
    """Delete partially generated per-image output and continue."""
    if image_dir.exists():
        try:
            shutil.rmtree(image_dir)
            print(f"[skip] {image_stem}: {reason}; removed {image_dir}")
        except Exception as exc:
            print(
                f"[warn] {image_stem}: failed to remove partial output {image_dir} "
                f"after error ({exc})"
            )
    else:
        print(f"[skip] {image_stem}: {reason}")


def _render_trajectory_overlay(
    *,
    image_dir: Path,
    image_stem: str,
    trajectory_root: Path,
    scene_meta_path: Optional[Path],
) -> bool:
    """Render per-image trajectory overlay PNG."""
    plot_script = Path(_PROJECT_ROOT) / "scripts" / "create_infer_sample" / "plot_trajectories_3d.py"
    if not plot_script.exists():
        print(f"[warn] {image_stem}: plot script missing, skip overlay -> {plot_script}")
        return False

    out_dir = _overlay_output_dir(image_dir, image_stem)
    plans_glob = str(trajectory_root / "*" / "plan.json")
    cmd = [
        sys.executable,
        str(plot_script),
        "--plans-glob",
        plans_glob,
        "--output-dir",
        str(out_dir),
        "--animate",
        "--animate-gif",
    ]
    if scene_meta_path is not None and scene_meta_path.exists():
        cmd.extend(["--scene-pointcloud", str(scene_meta_path)])

    try:
        subprocess.run(cmd, check=True)
    except Exception as exc:
        print(f"[warn] {image_stem}: failed to render trajectory overlay ({exc})")
        return False

    out_png = _overlay_png_path(image_dir, image_stem)
    return out_png.exists()


def _load_template_entries(trajectory_root: Path) -> List[Tuple[str, Dict, Path]]:
    """Load (combo_id, plan, template_dir) entries from template root."""
    entries: List[Tuple[str, Dict, Path]] = []
    for td in _find_plan_dirs(trajectory_root):
        plan_path = td / "plan.json"
        with plan_path.open("r", encoding="utf-8") as f:
            plan = json.load(f)
        combo_id = _extract_combo_id(td.name)
        entries.append((combo_id, plan, td))
    entries.sort(key=lambda x: x[0])
    return entries


def _expected_combo_ids(
    case_types: List[str],
    all_variants: bool,
) -> List[str]:
    """Resolve expected combo_ids from generate_eval_benchmark registry."""
    geb = _load_event_bench_module("generate_eval_benchmark")

    combo_ids = set()
    for case_type in case_types:
        combos = geb.get_combos_for_case(case_type, all_variants=all_variants)
        combo_ids.update(c.combo_id for c in combos)
    return sorted(combo_ids)


# ---------------------------------------------------------------------------
# Trajectory + scene meta generation (per-image mode)
# ---------------------------------------------------------------------------

def _prepare_image_to_target(image_rgb: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize/crop image to exact target size with minimal distortion."""
    from liveworld.geometry_utils import resize_short_edge_and_center_crop

    try:
        return resize_short_edge_and_center_crop(image_rgb, target_h, target_w)
    except ValueError:
        pass

    h, w = image_rgb.shape[:2]
    scale = min(target_w / float(w), target_h / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


class _Stream3RSceneBuilder:
    """Build first-frame scene pointcloud/meta with Stream3R and reuse one model."""

    def __init__(
        self,
        *,
        device: str,
        stream3r_model_path: str,
        target_h: int,
        target_w: int,
        voxel_size: float,
    ) -> None:
        from liveworld.geometry_utils import BackboneInferenceOptions
        from liveworld.pipelines.pointcloud_updater import create_pointcloud_updater

        self._BackboneInferenceOptions = BackboneInferenceOptions
        self._device = torch.device(device)
        self._stream3r_model_path = stream3r_model_path
        self._target_h = int(target_h)
        self._target_w = int(target_w)
        self._voxel_size = float(voxel_size)
        self._handler = create_pointcloud_updater("stream3r", self._device)

    def build(self, image_path: Path, output_dir: Path) -> Tuple[Path, Path]:
        from liveworld.geometry_utils import load_image, save_point_cloud_ply

        output_dir.mkdir(parents=True, exist_ok=True)

        image_rgb = load_image(str(image_path), rgb=True)
        frame = _prepare_image_to_target(image_rgb, self._target_h, self._target_w).astype(np.uint8)

        if hasattr(self._handler, "reset_session"):
            self._handler.reset_session()

        options = self._BackboneInferenceOptions(
            pointcloud_backend="stream3r",
            stream3r_model_path=self._stream3r_model_path,
            target_hw=(self._target_h, self._target_w),
            voxel_size=self._voxel_size,
        )
        geometry_poses = np.eye(4, dtype=np.float32)[None, ...]
        rec = self._handler.reconstruct_first_frame(
            frame=frame,
            geometry_poses_c2w=geometry_poses,
            dynamic_mask=None,
            options=options,
        )

        points_world = rec.points_world.astype(np.float32)
        colors = rec.colors if rec.colors is not None else np.full((len(points_world), 3), 255, dtype=np.uint8)
        colors = colors.astype(np.uint8)

        depth_hist = getattr(self._handler, "_depth_history", {})
        if 0 in depth_hist:
            depth_proc, c2w_hist, intrinsics_proc, hw = depth_hist[0]
            depth_proc = np.asarray(depth_proc, dtype=np.float32)
            intrinsics_proc = np.asarray(intrinsics_proc, dtype=np.float32)
            processed_size = np.array([int(hw[0]), int(hw[1])], dtype=np.int32)
            c2w = np.asarray(c2w_hist, dtype=np.float32)
        else:
            depth_proc = np.zeros((self._target_h, self._target_w), dtype=np.float32)
            intrinsics_proc = np.asarray(rec.intrinsics, dtype=np.float32)
            processed_size = np.array([self._target_h, self._target_w], dtype=np.int32)
            poses = self._handler.poses_c2w
            c2w = (
                np.asarray(poses[0], dtype=np.float32)
                if poses is not None and len(poses) > 0
                else np.eye(4, dtype=np.float32)
            )

        intrinsics = np.asarray(rec.intrinsics, dtype=np.float32)

        stem = image_path.stem
        ply_path = output_dir / f"{stem}_stream3r_scene.ply"
        meta_path = output_dir / f"{stem}_stream3r_scene_meta.npz"

        save_point_cloud_ply(ply_path, points_world, colors)
        np.savez_compressed(
            meta_path,
            points_world=points_world,
            colors=colors,
            intrinsics=intrinsics,
            c2w=c2w,
            target_h=int(self._target_h),
            target_w=int(self._target_w),
            depth_proc=depth_proc,
            intrinsics_proc=intrinsics_proc,
            processed_size=processed_size,
        )
        return ply_path, meta_path

    def cleanup(self) -> None:
        if self._handler is not None:
            self._handler.cleanup()
            self._handler = None


def _build_scene_pointcloud(
    image_path: Path,
    output_dir: Path,
    target_h: int,
    target_w: int,
    voxel_size: float,
) -> Tuple[Path, Path]:
    bspd = _load_event_bench_module("build_scene_pointcloud")
    return bspd.build_scene_pointcloud(
        image_path=str(image_path),
        output_dir=str(output_dir),
        target_h=int(target_h),
        target_w=int(target_w),
        voxel_size=float(voxel_size),
    )


def _generate_trajectory_templates_for_image(
    *,
    image_path: Path,
    scene_meta_path: Path,
    trajectory_root: Path,
    source_frames_per_round: int,
    qwen_model_path: str,
    sam3_model_path: str,
    system_config_path: Path,
    seed: int,
    case_types: List[str],
    all_variants: bool,
    llm_motion_scaling: bool,
    distance_mode: str,
    target_shift_ratio: float,
    screen_depth_percentile: float,
    distance_blend_alpha: float,
    exit_visible_ratio: float,
) -> List[Path]:
    """Generate per-image plan templates in-process."""
    trajectory_root.mkdir(parents=True, exist_ok=True)
    geb = _load_event_bench_module("generate_eval_benchmark")

    cli_args = [
        "--image-path", str(image_path),
        "--scene-meta-path", str(scene_meta_path),
        "--case-type", "case1",
        "--seed", str(seed),
        "--frames-per-round", str(source_frames_per_round),
        "--output-dir", str(trajectory_root),
        "--qwen-model-path", qwen_model_path,
        "--sam3-model-path", sam3_model_path,
        "--system-config-path", str(system_config_path),
        "--distance-mode", str(distance_mode),
        "--target-shift-ratio", str(float(target_shift_ratio)),
        "--screen-depth-percentile", str(float(screen_depth_percentile)),
        "--distance-blend-alpha", str(float(distance_blend_alpha)),
        "--exit-visible-ratio", str(float(exit_visible_ratio)),
    ]
    if all_variants:
        cli_args.append("--all-variants")
    if llm_motion_scaling:
        cli_args.append("--llm-motion-scaling")
    base_args = geb.build_argparser().parse_args(cli_args)

    scene_detect_prompt = geb._load_scene_detect_prompt(str(system_config_path))
    image_rgb = geb._load_image_rgb(str(image_path))
    scene_features = geb.extract_scene_features(image_rgb)
    scene_meta = geb._load_scene_meta(str(scene_meta_path))
    detect_image_path = geb._prepare_detection_image_for_scene(str(image_path), scene_meta)
    try:
        det, det_union_mask = geb.detect_event_object(
            image_path=detect_image_path,
            qwen_model_path=qwen_model_path,
            sam3_model_path=sam3_model_path,
            scene_detect_prompt=scene_detect_prompt,
        )
    except Exception as exc:
        print(
            f"[warn] {image_path.stem}: trajectory detection failed "
            f"({type(exc).__name__}: {exc})"
        )
        return []
    first_c2w = scene_meta["c2w"] if scene_meta is not None else np.eye(4, dtype=np.float32)

    try:
        for case_type in case_types:
            args = argparse.Namespace(**vars(base_args))
            args.case_type = case_type
            scene_scale = geb.estimate_scene_scale(
                det=det,
                scene_features=scene_features,
                image_path=str(image_path),
                geometry_path=getattr(args, "geometry_path", ""),
                case_type=args.case_type,
                llm_motion_scaling=args.llm_motion_scaling,
                qwen_model_path=args.qwen_model_path,
                scene_meta=scene_meta,
                distance_mode=args.distance_mode,
                target_shift_ratio=args.target_shift_ratio,
                screen_depth_percentile=args.screen_depth_percentile,
                distance_blend_alpha=args.distance_blend_alpha,
            )
            combos = geb.get_combos_for_case(
                args.case_type,
                all_variants=getattr(args, "all_variants", False),
            )
            for combo in combos:
                geb.generate_plan_for_combo(
                    combo=combo,
                    args=args,
                    image_rgb=image_rgb,
                    scene_features=scene_features,
                    det=det,
                    det_union_mask=det_union_mask,
                    scene_meta=scene_meta,
                    first_c2w=first_c2w,
                    scene_scale=scene_scale,
                )
    except Exception as exc:
        print(
            f"[warn] {image_path.stem}: trajectory template generation failed "
            f"({type(exc).__name__}: {exc})"
        )
        return []

    return _find_plan_dirs(trajectory_root)


# ---------------------------------------------------------------------------
# YAML writer with string-key quoting
# ---------------------------------------------------------------------------

class _QuotedStr(str):
    """String that PyYAML will always quote."""
    pass


def _quoted_str_representer(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="'")


yaml.add_representer(_QuotedStr, _quoted_str_representer)


def _write_combo_yaml(
    yaml_path: Path,
    geometry_rel: str,
    image_rel: str,
    entities_rel: str,
    storyline_rel: str,
    iter_input: Dict[str, Dict[str, str]],
    inference_output_root: str,
    image_stem: str,
    combo_id: str,
) -> None:
    """Write a per-combo inference YAML config."""
    quoted_iter_input = {}
    for k, v in iter_input.items():
        quoted_iter_input[_QuotedStr(str(k))] = v

    config = {
        "geometry_file_name": geometry_rel,
        "first_frame_image": image_rel,
        "entities_file": entities_rel,
        "storyline_file": storyline_rel,
        "iter_input": quoted_iter_input,
        "output": {"root": f"{inference_output_root}/{image_stem}/{combo_id}"},
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with yaml_path.open("w", encoding="utf-8") as f:
        f.write(f"# Auto-generated | combo: {combo_id} | image: {image_stem}\n")
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


# ---------------------------------------------------------------------------
# Entity detection
# ---------------------------------------------------------------------------

_DYNAMIC_FILTER_PROMPT = """\
Given this image and these detected objects: {entities}

Which of these can move ON THEIR OWN? (people, animals, birds, vehicles, robots, etc.)

STATIC (exclude): furniture, appliances, fixtures, radiators, doors, windows, \
lamps, plants, decorations, buildings, structures, signs, sky, water.

List ONLY the dynamic/movable ones. If none can move, output exactly: Nothing

Output format:
Nothing
OR numbered list:
1) person
2) dog"""


def _detect_entities(
    image_path: Path,
    qwen: Optional["Qwen3VLEntityExtractor"],
    scene_detect_prompt: str,
) -> Tuple[List[str], str]:
    """Detect entities from image using Qwen with scene_detect_prompt."""
    if qwen is None:
        return ["foreground entity"], "dry-run: no detection"
    detected, raw = qwen.extract(str(image_path), prompt=scene_detect_prompt)
    entities = sorted({str(e).strip() for e in detected if str(e).strip()})

    if entities:
        from liveworld.pipelines.monitor_centric.qwen_extractor import parse_entities
        filter_prompt = _DYNAMIC_FILTER_PROMPT.format(entities=", ".join(entities))
        filter_raw = qwen.generate_text(str(image_path), filter_prompt)
        filtered = parse_entities(filter_raw)
        original_lower = {e.lower() for e in entities}
        entities = [e for e in filtered if e.strip().lower() in original_lower]

    return entities, raw


# ---------------------------------------------------------------------------
# Storyline generation
# ---------------------------------------------------------------------------

def _generate_storyline(
    image_path: Path,
    entities: List[str],
    qwen: Optional["Qwen3VLEntityExtractor"],
    system_prompt_template: str,
    max_steps: int,
) -> Dict:
    """Generate shared event evolution storyline."""
    entities_text = ", ".join(entities) if entities else "foreground entity"
    primary_entity = entities[0] if entities else "foreground entity"

    def _fg_text_fallback(step_idx: int) -> str:
        return (
            f"{entities_text} walk toward a nearby object in the scene, moving a clear distance "
            "from their current position to interact with it while staying fully inside the frame. "
            f"(evolution step {step_idx + 1}/{max_steps})"
        )

    def _fg_text_violation_reason(fg_text: str) -> Optional[str]:
        lowered = fg_text.strip().lower()
        if not lowered:
            return "empty output"
        if any(re.search(p, lowered) for p in _FG_OUT_OF_FRAME_PATTERNS):
            return "mentions leaving the frame / off-screen motion"
        if any(re.search(p, lowered) for p in _FG_STATIONARY_MOTION_PATTERNS):
            return "describes stationary or tiny in-place movement"
        if any(re.search(p, lowered) for p in _FG_LOCAL_MOTION_PATTERNS):
            return "describes local-only micro motion"
        if not re.search(_FG_TRAVEL_MOTION_PATTERN, lowered):
            return "missing clear whole-body travel verb"
        if not any(re.search(p, lowered) for p in _FG_DISPLACEMENT_CUE_PATTERNS):
            return "missing explicit start-to-destination displacement cue"
        if not any(re.search(p, lowered) for p in _FG_IN_FRAME_CUE_PATTERNS):
            return "missing explicit in-frame constraint"
        return None

    if qwen is not None:
        scene_prompt = _SCENE_TEXT_PROMPT.format(
            entities=entities_text,
            trajectory_description="The camera is stationary at its initial position.",
        )
        scene_text = qwen.generate_text(str(image_path), prompt=scene_prompt).strip()
        if scene_text.startswith("{") or scene_text.startswith('"'):
            scene_text = scene_text.strip('"{}').strip()
        if _scene_text_violates_rules(scene_text):
            scene_text = _scene_text_fallback()
    else:
        scene_text = _scene_text_fallback()

    steps: Dict[str, str] = {}
    prev_prompt = ""

    for step_idx in range(max_steps):
        instruction = system_prompt_template.format(
            entities=entities_text,
            entity=primary_entity,
            iteration=step_idx,
        )
        instruction += "\n\n" + _FG_MOTION_HARD_RULES
        if prev_prompt:
            instruction += (
                f"\n\nPrevious evolution description: {prev_prompt}\n"
                "Continue naturally from where the previous evolution ended. "
                "The entities must move to a DIFFERENT object or location in the scene — "
                "approach a bench, walk toward a tree, trot over to a ball, head to a doorway, etc. "
                "Prioritize INTERACTING with visible objects in the scene rather than aimless wandering. "
                "The viewer should see them end up in a visibly different spot than where they started. "
                "Do NOT let them stay in the same place, stand still, or only fidget. "
                "IMPORTANT: all entities must stay fully inside the frame at all times — "
                "never walk out of frame or move to the edge. "
                "Do not repeat the previous description."
            )
        instruction += (
            "\n\nWrite one compact motion sentence with explicit destination and frame safety, for example:\n"
            "'<entity> walks from the left side toward the bench on the right while staying fully inside the frame.'\n"
            "'<entity> strolls over to the flower pot near the wall and sniffs it, remaining inside the frame.'"
        )

        if qwen is not None:
            fg_text = ""
            violation_reason: Optional[str] = "empty output"
            for attempt_idx in range(_FG_REWRITE_MAX_ATTEMPTS):
                prompt = instruction
                if attempt_idx > 0:
                    prompt += (
                        "\n\nYour previous draft was invalid because it "
                        f"{violation_reason}. Rewrite and strictly satisfy all motion rules."
                    )
                fg_text = qwen.generate_text(str(image_path), prompt=prompt).strip()
                if fg_text.startswith("{") or fg_text.startswith('"'):
                    fg_text = fg_text.strip('"{}').strip()
                violation_reason = _fg_text_violation_reason(fg_text)
                if violation_reason is None:
                    break
            if violation_reason is not None:
                fg_text = _fg_text_fallback(step_idx)
        else:
            fg_text = _fg_text_fallback(step_idx)

        steps[str(step_idx)] = fg_text.strip()
        prev_prompt = fg_text.strip()

    return {
        "image_stem": image_path.stem,
        "entities": entities,
        "max_steps": max_steps,
        "scene_text": scene_text,
        "steps": steps,
        "generated_at": datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# iter_input assembly
# ---------------------------------------------------------------------------

_SCENE_TEXT_PROMPT = """
You are labeling scene descriptions for an event-centric video benchmark.
The source image shows the scene from the initial view. The current iteration
corresponds to a different point along a pre-defined trajectory. Based on the
image, IMAGINE and describe what the static environment would look like now.

Foreground entities to EXCLUDE (do NOT mention these at all): {entities}

Trajectory state for this iteration:
{trajectory_description}

Rules:
- Describe only visible static-environment content in the current frame.
- Focus on layout/depth/occlusion changes as the viewpoint travels.
- Describe ONLY static environment: architecture, terrain, sky, vegetation,
  furniture, floor, walls, lighting, etc.
- Do NOT mention any animals, people, or dynamic foreground objects.
- Do NOT use control words like forward/backward/left/right/cw/ccw.
- Do NOT say "camera faces/turns/points ...".
- Keep it concise (1-2 sentences).
- Output the scene_text string only, no JSON wrapping.
""".strip()


def _yaw_from_c2w(c2w: List[List[float]]) -> float:
    """Extract yaw angle (degrees) from c2w."""
    fwd_x = c2w[0][2]
    fwd_z = c2w[2][2]
    return math.degrees(math.atan2(fwd_x, fwd_z))


def _describe_trajectory_state(
    round_info: Dict,
    phase_idx: int,
    phase_count: int,
) -> str:
    """Build a direction-agnostic trajectory description at a given phase."""
    t = phase_idx / max(phase_count - 1, 1)

    start_pos = round_info["camera_start"]
    end_pos = round_info["camera_end"]
    pos = [s + t * (e - s) for s, e in zip(start_pos, end_pos)]

    start_c2w = round_info["pose_start_c2w"]
    end_c2w = round_info["pose_end_c2w"]
    interp_c2w = [
        [s + t * (e - s) for s, e in zip(row_s, row_e)]
        for row_s, row_e in zip(start_c2w, end_c2w)
    ]

    yaw = _yaw_from_c2w(interp_c2w)
    start_yaw = _yaw_from_c2w(start_c2w)
    end_yaw = _yaw_from_c2w(end_c2w)
    yaw_delta = end_yaw - start_yaw

    dx = end_pos[0] - start_pos[0]
    dz = end_pos[2] - start_pos[2]
    move_dist = math.sqrt(dx * dx + dz * dz)

    action = round_info["action"]
    loc_from = round_info["location_from"]
    loc_to = round_info["location_to"]

    lines = []
    lines.append(f"- Motion action: {action}; path segment: {loc_from}->{loc_to}")
    lines.append(f"- Segment phase: {phase_idx + 1}/{phase_count}")
    lines.append(f"- Position offset (X,Z): ({pos[0]:.2f}, {pos[2]:.2f})")
    lines.append(f"- Segment translation magnitude: {move_dist:.2f}")
    lines.append(f"- Segment yaw change magnitude: {abs(yaw_delta):.1f} deg")
    lines.append(f"- Current yaw value: {yaw:.1f} deg")

    if phase_idx == 0:
        lines.append("- This is the beginning of the segment")
    elif phase_idx == phase_count - 1:
        lines.append("- This is the end of the segment")

    return "\n".join(lines)


def _scene_text_violates_rules(scene_text: str) -> bool:
    lowered = scene_text.strip().lower()
    if not lowered:
        return True
    return any(re.search(p, lowered) for p in _SCENE_CAMERA_EGO_PATTERNS)


def _scene_text_fallback() -> str:
    return (
        "The static environment remains coherent, with background layers and "
        "occlusion boundaries shifting smoothly along the trajectory."
    )


def _build_combo_iter_input(
    plan: Dict,
    phase_count: int,
    entities: List[str],
    qwen: Optional["Qwen3VLEntityExtractor"],
    image_path: Path,
    storyline_steps: Dict[str, str],
) -> Dict[str, Dict[str, str]]:
    """Build runtime iter_input for one combo."""
    num_rounds = plan["num_rounds"]
    round_plan = plan["round_plan"]
    total_iters = num_rounds * phase_count
    entities_text = ", ".join(entities) if entities else "any foreground objects"

    iter_input: Dict[str, Dict[str, str]] = {}

    for iter_idx in range(total_iters):
        internal_round_idx = iter_idx // phase_count
        phase_idx = iter_idx % phase_count
        round_info = round_plan[str(internal_round_idx)]

        trajectory_description = _describe_trajectory_state(round_info, phase_idx, phase_count)

        if qwen is not None:
            prompt = _SCENE_TEXT_PROMPT.format(
                entities=entities_text,
                trajectory_description=trajectory_description,
            )
            scene_text = qwen.generate_text(str(image_path), prompt=prompt).strip()
            if scene_text.startswith("{") or scene_text.startswith('"'):
                scene_text = scene_text.strip('"{}').strip()
            if _scene_text_violates_rules(scene_text):
                scene_text = _scene_text_fallback()
        else:
            scene_text = _scene_text_fallback()

        iter_input[str(iter_idx)] = {
            "scene_text": scene_text,
            "fg_text": "",
        }

    return iter_input


# ---------------------------------------------------------------------------
# Main assembly
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble per-image event bench folders (case1 + case2)."
    )
    parser.add_argument("--images-dir", default=None)
    parser.add_argument("--images-list", default=None)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--source-frames-per-round", type=int, default=66)
    parser.add_argument("--target-frames-per-iter", type=int, default=33)
    parser.add_argument("--qwen-model-path", default="ckpts/Qwen--Qwen3-VL-8B-Instruct")
    parser.add_argument("--sam3-model-path", default="ckpts/facebook--sam3/sam3.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--system-config-path",
        default="configs/infer_system_config.yaml",
    )
    parser.add_argument("--max-storyline-steps", type=int, default=8)
    parser.add_argument("--trajectory-case-types", default="case1,case2")
    parser.add_argument("--trajectory-all-variants", action="store_true")
    parser.add_argument("--trajectory-llm-motion-scaling", action="store_true")
    parser.add_argument(
        "--trajectory-distance-mode", default="screen_uniform",
        choices=["screen_uniform"],
    )
    parser.add_argument("--trajectory-target-shift-ratio", type=float, default=0.18)
    parser.add_argument("--trajectory-screen-depth-percentile", type=float, default=45.0)
    parser.add_argument("--trajectory-distance-blend-alpha", type=float, default=0.7)
    parser.add_argument("--trajectory-exit-visible-ratio", type=float, default=0.01)
    parser.add_argument("--scene-estimator", default="stream3r", choices=["stream3r", "da3"])
    parser.add_argument("--stream3r-model-path", default="ckpts/yslan--STream3R")
    parser.add_argument("--target-height", type=int, default=480)
    parser.add_argument("--target-width", type=int, default=832)
    parser.add_argument("--scene-voxel-size", type=float, default=0.01)
    parser.add_argument("--inference-output-root", default="outputs_event_bench")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-trajectory-overlay", dest="save_trajectory_overlay",
                        action="store_true", default=True)
    parser.add_argument("--no-save-trajectory-overlay", dest="save_trajectory_overlay",
                        action="store_false")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    output_root = Path(args.output_root).resolve()
    system_config_path = Path(args.system_config_path).resolve()

    if not system_config_path.exists():
        raise FileNotFoundError(f"system config not found: {system_config_path}")

    if args.images_list:
        images_list_path = Path(args.images_list)
        if not images_list_path.exists():
            raise FileNotFoundError(f"images list not found: {images_list_path}")
        image_paths = [
            Path(line.strip()) for line in images_list_path.read_text().splitlines()
            if line.strip()
        ]
    elif args.images_dir:
        images_dir = Path(args.images_dir).resolve()
        if not images_dir.exists():
            raise FileNotFoundError(f"images dir not found: {images_dir}")
        image_paths = _find_image_files(images_dir)
    else:
        raise ValueError("Must provide either --images-dir or --images-list")

    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise FileNotFoundError("No images found")
    print(f"[assemble] Found {len(image_paths)} images")

    if args.source_frames_per_round % args.target_frames_per_iter != 0:
        raise ValueError(
            f"source_frames_per_round={args.source_frames_per_round} not divisible by "
            f"target_frames_per_iter={args.target_frames_per_iter}"
        )
    phase_count = args.source_frames_per_round // args.target_frames_per_iter
    print(f"[assemble] phase_count={phase_count}")

    case_types = [c.strip() for c in args.trajectory_case_types.split(",") if c.strip()]
    invalid = [c for c in case_types if c not in {"case1", "case2"}]
    if invalid:
        raise ValueError(f"Invalid --trajectory-case-types: {invalid}")

    qwen = None
    if not args.dry_run:
        Qwen3VLEntityExtractor = _load_qwen_extractor_class()
        print(f"[qwen] loading model from {args.qwen_model_path} on {args.device}")
        qwen = Qwen3VLEntityExtractor(model_path=args.qwen_model_path, device=args.device)

    system_prompt_template, scene_detect_prompt = _load_event_prompts(str(system_config_path))

    expected_combo_ids = _expected_combo_ids(
        case_types=case_types,
        all_variants=args.trajectory_all_variants,
    )
    if not expected_combo_ids:
        raise RuntimeError("No trajectory combos resolved for current case settings.")
    print(
        f"[assemble] per-image trajectory mode: expected combos={len(expected_combo_ids)} "
        f"({', '.join(expected_combo_ids)})"
    )

    output_root.mkdir(parents=True, exist_ok=True)

    scene_builder = None
    if args.scene_estimator == "stream3r":
        print(
            f"[scene] loading Stream3R scene builder: {args.stream3r_model_path} "
            f"on {args.device}"
        )
        scene_builder = _Stream3RSceneBuilder(
            device=args.device,
            stream3r_model_path=args.stream3r_model_path,
            target_h=args.target_height,
            target_w=args.target_width,
            voxel_size=args.scene_voxel_size,
        )

    total_combos = 0
    try:
        for image_path in image_paths:
            image_stem = image_path.stem
            image_dir = output_root / image_stem
            scripts_dir = image_dir / "infer_scripts"
            scene_pc_dir = image_dir / "scene_pointcloud"
            traj_dir = image_dir / "trajectory_templates"
            entities_path = image_dir / "entities.json"
            storyline_path = image_dir / "storyline.json"
            status_path = image_dir / "assemble_status.json"
            dest_image = image_dir / f"source_image{image_path.suffix}"
            overlay_png_file = _overlay_png_path(image_dir, image_stem)
            source_image_exists = (
                dest_image.exists()
                or bool(list(image_dir.glob("source_image.*")))
            )

            existing_yaml_ids = (
                {p.stem for p in scripts_dir.glob("*.yaml")}
                if scripts_dir.exists()
                else set()
            )
            completed_base = (
                bool(expected_combo_ids)
                and set(expected_combo_ids).issubset(existing_yaml_ids)
                and source_image_exists
                and entities_path.exists()
                and storyline_path.exists()
            )
            overlay_ready = (not args.save_trajectory_overlay) or overlay_png_file.exists()
            completed = completed_base and overlay_ready

            if (
                args.resume
                and not args.overwrite
                and completed_base
                and args.save_trajectory_overlay
                and not overlay_ready
            ):
                scene_meta_path = _find_scene_meta_path(scene_pc_dir)
                if traj_dir.exists():
                    overlay_ok = _render_trajectory_overlay(
                        image_dir=image_dir,
                        image_stem=image_stem,
                        trajectory_root=traj_dir,
                        scene_meta_path=scene_meta_path,
                    )
                    if overlay_ok:
                        print(f"[resume] {image_stem}: generated missing trajectory overlay")
                    else:
                        print(f"[warn] {image_stem}: failed to backfill trajectory overlay")
                overlay_ready = overlay_png_file.exists()
                completed = completed_base and overlay_ready

            if args.resume and not args.overwrite and completed:
                print(
                    f"[skip] {image_stem}: already complete "
                    f"({len(expected_combo_ids)}/{len(expected_combo_ids)} combos)"
                )
                total_combos += len(expected_combo_ids)
                continue
            if args.resume and not args.overwrite and existing_yaml_ids:
                print(
                    f"[resume] {image_stem}: found partial infer scripts "
                    f"({len(existing_yaml_ids)}/{len(expected_combo_ids)} combos)"
                )

            image_dir.mkdir(parents=True, exist_ok=True)
            scripts_dir.mkdir(parents=True, exist_ok=True)

            # 1) Copy source image.
            if args.resume and not args.overwrite:
                existing_source_images = sorted(image_dir.glob("source_image.*"))
                if not dest_image.exists() and existing_source_images:
                    dest_image = existing_source_images[0]
                if dest_image.exists():
                    print(f"[resume] {image_stem}: reuse source image -> {dest_image.name}")
                else:
                    shutil.copy2(str(image_path), str(dest_image))
            else:
                shutil.copy2(str(image_path), str(dest_image))

            # 2) Detect entities.
            entities_data: Optional[Dict] = None
            if args.resume and not args.overwrite and entities_path.exists():
                try:
                    cached = _read_json(entities_path)
                    if isinstance(cached, dict) and isinstance(cached.get("entities"), list):
                        entities_data = cached
                        print(f"[resume] {image_stem}: reuse entities.json")
                except Exception as exc:
                    print(f"[resume] {image_stem}: entities.json invalid ({exc}), regenerating")
            if entities_data is None:
                try:
                    entities, entity_raw = _detect_entities(image_path, qwen, scene_detect_prompt)
                except Exception as exc:
                    print(
                        f"[warn] {image_stem}: entity detection failed "
                        f"({type(exc).__name__}: {exc}), using fallback entity"
                    )
                    entities = ["foreground entity"]
                    entity_raw = f"fallback (detection error: {exc})"
                if not entities:
                    entities = ["foreground entity"]
                entities_data = {
                    "entities": entities,
                    "entity_detect_raw": entity_raw,
                    "primary_entity": entities[0] if entities else "",
                    "image_path": str(image_path),
                }
                _write_json(entities_path, entities_data)
            entities = [str(e).strip() for e in entities_data.get("entities", []) if str(e).strip()]
            print(f"[entity] {image_stem}: {', '.join(entities) if entities else 'None'}")

            # 3) Generate storyline.
            storyline: Optional[Dict] = None
            if args.resume and not args.overwrite and storyline_path.exists():
                try:
                    cached_storyline = _read_json(storyline_path)
                    cached_steps = (
                        cached_storyline.get("steps", {})
                        if isinstance(cached_storyline, dict)
                        else {}
                    )
                    cached_max_steps = (
                        int(cached_storyline.get("max_steps", -1))
                        if isinstance(cached_storyline, dict)
                        else -1
                    )
                    if isinstance(cached_storyline, dict) and isinstance(cached_steps, dict):
                        if cached_max_steps == int(args.max_storyline_steps):
                            storyline = cached_storyline
                            print(f"[resume] {image_stem}: reuse storyline.json ({cached_max_steps} steps)")
                        else:
                            print(f"[resume] {image_stem}: storyline max_steps mismatch, regenerating")
                except Exception as exc:
                    print(f"[resume] {image_stem}: storyline.json invalid ({exc}), regenerating")
            if storyline is None:
                try:
                    storyline = _generate_storyline(
                        image_path=image_path,
                        entities=entities,
                        qwen=qwen,
                        system_prompt_template=system_prompt_template,
                        max_steps=args.max_storyline_steps,
                    )
                except Exception as exc:
                    print(
                        f"[warn] {image_stem}: storyline generation failed "
                        f"({type(exc).__name__}: {exc}), using placeholder"
                    )
                    entities_text = ", ".join(entities) if entities else "foreground entity"
                    storyline = {
                        "scene_text": "A scene with objects in the environment.",
                        "max_steps": int(args.max_storyline_steps),
                        "steps": {
                            str(i): (
                                f"{entities_text} walks steadily forward while staying "
                                "fully inside the frame."
                            )
                            for i in range(int(args.max_storyline_steps))
                        },
                    }
                _write_json(storyline_path, storyline)
                print(f"[storyline] {image_stem}: {storyline['max_steps']} steps generated")

            # 4) Resolve templates for this image.
            if args.overwrite and scene_pc_dir.exists():
                shutil.rmtree(scene_pc_dir)
            if args.overwrite and traj_dir.exists():
                shutil.rmtree(traj_dir)
            scene_pc_dir.mkdir(parents=True, exist_ok=True)
            traj_dir.mkdir(parents=True, exist_ok=True)

            scene_meta_path: Optional[Path] = None
            template_entries = []
            templates_complete = False
            if args.resume and not args.overwrite:
                scene_meta_path = _find_scene_meta_path(scene_pc_dir)
                if scene_meta_path is not None:
                    print(f"[resume] {image_stem}: reuse scene meta -> {scene_meta_path.name}")
                template_entries = _load_template_entries(traj_dir)
                current_template_ids = {combo_id for combo_id, _, _ in template_entries}
                templates_complete = (
                    bool(expected_combo_ids)
                    and set(expected_combo_ids).issubset(current_template_ids)
                )
                if templates_complete:
                    print(
                        f"[resume] {image_stem}: reuse trajectory templates "
                        f"({len(current_template_ids)}/{len(expected_combo_ids)} combos)"
                    )

            if scene_meta_path is None:
                try:
                    if args.scene_estimator == "stream3r":
                        if scene_builder is None:
                            raise RuntimeError("Stream3R scene builder is not initialized")
                        _, scene_meta_path = scene_builder.build(
                            image_path=image_path,
                            output_dir=scene_pc_dir,
                        )
                    else:
                        _, scene_meta_path = _build_scene_pointcloud(
                            image_path=image_path,
                            output_dir=scene_pc_dir,
                            target_h=args.target_height,
                            target_w=args.target_width,
                            voxel_size=args.scene_voxel_size,
                        )
                except Exception as exc:
                    _cleanup_failed_image_output(
                        image_dir=image_dir,
                        image_stem=image_stem,
                        reason=f"scene reconstruction failed ({type(exc).__name__}: {exc})",
                    )
                    continue
                print(f"[scene] {image_stem}: scene meta -> {scene_meta_path}")

            if not templates_complete:
                template_dirs = _generate_trajectory_templates_for_image(
                    image_path=image_path,
                    scene_meta_path=Path(scene_meta_path),
                    trajectory_root=traj_dir,
                    source_frames_per_round=args.source_frames_per_round,
                    qwen_model_path=args.qwen_model_path,
                    sam3_model_path=args.sam3_model_path,
                    system_config_path=system_config_path,
                    seed=args.seed,
                    case_types=case_types,
                    all_variants=args.trajectory_all_variants,
                    llm_motion_scaling=args.trajectory_llm_motion_scaling,
                    distance_mode=args.trajectory_distance_mode,
                    target_shift_ratio=args.trajectory_target_shift_ratio,
                    screen_depth_percentile=args.trajectory_screen_depth_percentile,
                    distance_blend_alpha=args.trajectory_distance_blend_alpha,
                    exit_visible_ratio=args.trajectory_exit_visible_ratio,
                )
                if not template_dirs:
                    _cleanup_failed_image_output(
                        image_dir=image_dir,
                        image_stem=image_stem,
                        reason="trajectory generation failed (no templates produced)",
                    )
                    continue
                template_entries = _load_template_entries(traj_dir)
                current_template_ids = {combo_id for combo_id, _, _ in template_entries}
                if not set(expected_combo_ids).issubset(current_template_ids):
                    _cleanup_failed_image_output(
                        image_dir=image_dir,
                        image_stem=image_stem,
                        reason=(
                            "trajectory generation failed (templates incomplete "
                            f"{len(current_template_ids)}/{len(expected_combo_ids)})"
                        ),
                    )
                    continue
                print(
                    f"[trajectory] {image_stem}: generated {len(template_entries)} templates "
                    f"(estimator={args.scene_estimator})"
                )

            if args.save_trajectory_overlay:
                if args.overwrite or not overlay_png_file.exists():
                    overlay_ok = _render_trajectory_overlay(
                        image_dir=image_dir,
                        image_stem=image_stem,
                        trajectory_root=traj_dir,
                        scene_meta_path=scene_meta_path,
                    )
                    if overlay_ok:
                        print(f"[trajectory-viz] {image_stem}: {overlay_png_file}")
                    else:
                        print(f"[warn] {image_stem}: trajectory overlay missing after render")
                else:
                    print(f"[resume] {image_stem}: reuse trajectory overlay")

            # 5) Generate per-combo YAML configs.
            bench_root = output_root.parent
            generated_yaml_count = 0
            reused_yaml_count = 0
            for combo_id, template_plan, template_dir in template_entries:
                yaml_path = scripts_dir / f"{combo_id}.yaml"
                if args.resume and not args.overwrite and yaml_path.exists():
                    reused_yaml_count += 1
                    total_combos += 1
                    continue
                geom_src = template_dir / "geometry.npz"
                if not geom_src.exists():
                    print(f"[warn] {image_stem}/{combo_id}: missing geometry.npz, skip")
                    continue

                iter_input = _build_combo_iter_input(
                    plan=template_plan,
                    phase_count=phase_count,
                    entities=entities,
                    qwen=qwen,
                    image_path=image_path,
                    storyline_steps=storyline["steps"],
                )
                geometry_rel = str(geom_src.relative_to(bench_root))
                image_rel = str(dest_image.relative_to(bench_root))
                entities_rel = str(entities_path.relative_to(bench_root))
                storyline_rel = str(storyline_path.relative_to(bench_root))

                _write_combo_yaml(
                    yaml_path=yaml_path,
                    geometry_rel=geometry_rel,
                    image_rel=image_rel,
                    entities_rel=entities_rel,
                    storyline_rel=storyline_rel,
                    iter_input=iter_input,
                    inference_output_root=args.inference_output_root,
                    image_stem=image_stem,
                    combo_id=combo_id,
                )
                generated_yaml_count += 1
                total_combos += 1

            final_yaml_ids = {p.stem for p in scripts_dir.glob("*.yaml")}
            image_completed = (
                bool(expected_combo_ids)
                and set(expected_combo_ids).issubset(final_yaml_ids)
            )
            _write_json(
                status_path,
                {
                    "image_stem": image_stem,
                    "image_path": str(image_path),
                    "completed": image_completed,
                    "expected_combos": expected_combo_ids,
                    "yaml_count": len(final_yaml_ids),
                    "generated_yaml_count": generated_yaml_count,
                    "reused_yaml_count": reused_yaml_count,
                    "trajectory_overlay_ready": bool(
                        (not args.save_trajectory_overlay) or overlay_png_file.exists()
                    ),
                    "resume_enabled": bool(args.resume and not args.overwrite),
                    "updated_at": datetime.now().isoformat(),
                },
            )
            print(
                f"[ok] {image_stem}: generated={generated_yaml_count}, reused={reused_yaml_count}, "
                f"done={len(final_yaml_ids)}/{len(expected_combo_ids)} combos "
                f"({len(entities)} entities, {storyline['max_steps']} storyline steps)"
            )
    finally:
        if scene_builder is not None:
            scene_builder.cleanup()

    print(f"\n[done] output_root={output_root}")
    print(f"[done] images={len(image_paths)}, combos_total={total_combos}")


if __name__ == "__main__":
    main()
