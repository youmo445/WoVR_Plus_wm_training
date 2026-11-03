#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

from .projection import (
    generate_raymap,
    generate_traj_map,
    EndEffectorPts,
    Gripper2EEFCvt,
    get_transformation_matrix_from_quat,
    to_uint8_img,
    write_video_rgb,
)

ROBOTWIN_THREE_VIEW_CAMERAS = ("head_camera", "left_camera", "right_camera")
ROBOTWIN_ALL_CAMERAS = ("front_camera", "head_camera", "left_camera", "right_camera")


def _episode_sort_key(path: Path) -> Tuple[int, str]:
    match = re.search(r"episode(\d+)", path.stem)
    if match:
        return int(match.group(1)), path.name
    return 10**9, path.name


def _decode_rgb_bytes(raw) -> np.ndarray:
    if isinstance(raw, np.bytes_):
        raw = raw.tobytes()
    elif isinstance(raw, np.ndarray):
        raw = raw.tobytes()
    elif not isinstance(raw, (bytes, bytearray)):
        raw = bytes(raw)
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)


def _to_4x4_w2c(extrinsic_cv: np.ndarray) -> np.ndarray:
    extrinsic_cv = np.asarray(extrinsic_cv, dtype=np.float32)
    if extrinsic_cv.shape == (4, 4):
        return extrinsic_cv
    if extrinsic_cv.shape != (3, 4):
        raise ValueError(f"Expected extrinsic_cv shape (3, 4) or (4, 4), got {extrinsic_cv.shape}")
    out = np.eye(4, dtype=np.float32)
    out[:3, :] = extrinsic_cv
    return out


def _scale_intrinsic(
    intrinsic: np.ndarray, in_h: int, in_w: int, out_h: int, out_w: int
) -> np.ndarray:
    K = np.asarray(intrinsic, dtype=np.float32).copy()
    K[0, 0] *= out_w / float(in_w)
    K[0, 2] *= out_w / float(in_w)
    K[1, 1] *= out_h / float(in_h)
    K[1, 2] *= out_h / float(in_h)
    return K


def _project_sphere_to_image(
    center_world: np.ndarray,
    radius_m: float,
    w2c: np.ndarray,
    K: np.ndarray,
    image_height: int,
    image_width: int,
    min_radius_px: float,
    max_radius_px: float,
) -> Optional[Tuple[float, float, float]]:
    center_h = np.concatenate(
        [np.asarray(center_world, dtype=np.float32), np.ones(1, dtype=np.float32)]
    )
    center_cam = np.asarray(w2c, dtype=np.float32) @ center_h
    depth = float(center_cam[2])
    if depth <= 1e-6:
        return None
    uvw = np.asarray(K, dtype=np.float32) @ center_cam[:3]
    if abs(float(uvw[2])) <= 1e-6:
        return None
    u = float(uvw[0] / uvw[2])
    v = float(uvw[1] / uvw[2])
    focal = 0.5 * (float(K[0, 0]) + float(K[1, 1]))
    radius_px = focal * float(radius_m) / depth
    radius_px = float(np.clip(radius_px, min_radius_px, max_radius_px))
    if (
        u + radius_px < 0
        or u - radius_px >= image_width
        or v + radius_px < 0
        or v - radius_px >= image_height
    ):
        return None
    return u, v, radius_px


def _blur_projected_sphere_region(
    image: Image.Image,
    center_world: np.ndarray,
    radius_m: float,
    w2c: np.ndarray,
    K: np.ndarray,
    sigma_px: float,
    min_radius_px: float,
    max_radius_px: float,
) -> Image.Image:
    width, height = image.size
    projected = _project_sphere_to_image(
        center_world, radius_m, w2c, K, height, width, min_radius_px, max_radius_px
    )
    if projected is None:
        return image
    u, v, radius_px = projected
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((u - radius_px, v - radius_px, u + radius_px, v + radius_px), fill=255)
    feather_px = max(1.0, radius_px * 0.18)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_px))
    blurred = image.filter(ImageFilter.GaussianBlur(radius=float(sigma_px)))
    return Image.composite(blurred, image, mask)


def _eef_projection_center_world(abs_action: np.ndarray, arm_start: int) -> np.ndarray:
    pose = get_transformation_matrix_from_quat(
        np.asarray(abs_action[None, arm_start : arm_start + 7], dtype=np.float32)
    )[0]
    cvt_matrix = Gripper2EEFCvt.reshape(4, 4)
    ee_key_pts = EndEffectorPts.reshape(1, 4, 4).transpose(0, 2, 1)[0]
    center_h = pose @ cvt_matrix @ ee_key_pts[:, 0]
    if abs(float(center_h[3])) > 1e-8:
        center_h = center_h / center_h[3]
    return np.asarray(center_h[:3], dtype=np.float32)


def _make_abs_actions_from_endpose(
    left_endpose: np.ndarray,
    right_endpose: np.ndarray,
    left_gripper: np.ndarray,
    right_gripper: np.ndarray,
    quat_order: str,
) -> np.ndarray:
    left_endpose = np.asarray(left_endpose, dtype=np.float32)
    right_endpose = np.asarray(right_endpose, dtype=np.float32)
    out = np.zeros((left_endpose.shape[0], 16), dtype=np.float32)
    out[:, 0:3] = left_endpose[:, 0:3]
    out[:, 8:11] = right_endpose[:, 0:3]
    if quat_order == "xyzw":
        out[:, 3:7] = left_endpose[:, 3:7]
        out[:, 11:15] = right_endpose[:, 3:7]
    elif quat_order == "wxyz":
        out[:, 3:7] = left_endpose[:, [4, 5, 6, 3]]
        out[:, 11:15] = right_endpose[:, [4, 5, 6, 3]]
    else:
        raise ValueError(f"Unsupported quat_order: {quat_order}")
    out[:, 7] = np.clip(np.asarray(left_gripper, dtype=np.float32), 0.0, 1.0) * 120.0
    out[:, 15] = np.clip(np.asarray(right_gripper, dtype=np.float32), 0.0, 1.0) * 120.0
    return out


class RoboTwinDatasetWancontrolmultiview(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path: str = "../wovr-assets/datasets/adjust_bottle",
        num_frames: int = 13,
        repeat: int = 1,
        split: str = "train",
        train_episodes_per_task: int = 40,
        val_episodes_per_task: int = 10,
        stride: int = 1,
        first_round_prob: float = 0.05,
        context_length: int = 5,
        sample_mode: str = "contiguous",
        sparse_history_stride: int = 4,
        temporal_stride_choices: Sequence[int] = (1, 2, 3, 4),
        temporal_stride_probs: Sequence[float] = (0.5, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0),
        context_sphere_blur_prob: float = 0.0,
        context_sphere_blur_camera: str = "random",
        context_sphere_blur_radius_min: float = 0.04,
        context_sphere_blur_radius_max: float = 0.08,
        context_sphere_blur_sigma_min: float = 3.0,
        context_sphere_blur_sigma_max: float = 6.0,
        context_sphere_blur_min_radius_px: float = 10.0,
        context_sphere_blur_max_radius_px: float = 42.0,
        context_sphere_blur_position_noise_std: float = 0.025,
        traj_radius: int = 40,
        traj_radius_mode: str = "perspective",
        traj_min_radius: int = 14,
        traj_max_radius: int = 58,
        traj_ref_depth: float = 0.30,
        traj_near_depth: float = 0.19,
        traj_far_depth: float = 0.69,
        output_raymap: bool = False,
        ray_o_vmin: float = -1.5,
        ray_o_vmax: float = 1.5,
        ray_d_vmin: float = -1.0,
        ray_d_vmax: float = 1.0,
        raymap_mode: str = "image",
        resize_to: Optional[Tuple[int, int]] = None,
        dataset_type: str = "vace",
        view_mode: str = "three",
        single_camera: str = "head_camera",
        camera_names: Optional[Sequence[str]] = None,
        endpose_quat_order: str = "wxyz",
        prompt: str = "机械臂按照要求移动夹爪执行任务",
        task_limit: Optional[int] = None,
    ):
        super().__init__()
        if split not in {"train", "val", "all"}:
            raise ValueError(f"Unsupported split: {split}")
        if dataset_type not in {"vace", "control"}:
            raise ValueError(f"Unsupported dataset_type: {dataset_type}")
        if raymap_mode != "image":
            raise ValueError(
                "RoboTwinDatasetWancontrolmultiview currently supports raymap_mode='image'"
            )
        if view_mode not in {"three", "single"}:
            raise ValueError(f"Unsupported view_mode: {view_mode}")
        if single_camera not in ROBOTWIN_ALL_CAMERAS:
            raise ValueError(f"Unsupported single_camera: {single_camera}")
        if endpose_quat_order not in {"wxyz", "xyzw"}:
            raise ValueError(f"Unsupported endpose_quat_order: {endpose_quat_order}")
        if sample_mode not in {"contiguous", "fixed_sparse_history", "random_temporal_stride"}:
            raise ValueError(f"Unsupported sample_mode: {sample_mode}")
        if sample_mode == "fixed_sparse_history" and int(context_length) != 5:
            raise ValueError("fixed_sparse_history expects context_length=5.")
        if int(sparse_history_stride) <= 0:
            raise ValueError(f"sparse_history_stride must be positive, got {sparse_history_stride}")
        if isinstance(temporal_stride_choices, str):
            temporal_stride_choices = [
                x.strip() for x in temporal_stride_choices.split(",") if x.strip()
            ]
        if isinstance(temporal_stride_probs, str):
            temporal_stride_probs = [
                x.strip() for x in temporal_stride_probs.split(",") if x.strip()
            ]
        temporal_stride_choices = [int(x) for x in temporal_stride_choices]
        temporal_stride_probs = np.asarray(
            [float(x) for x in temporal_stride_probs], dtype=np.float64
        )
        if len(temporal_stride_choices) == 0:
            raise ValueError("temporal_stride_choices cannot be empty.")
        if len(temporal_stride_choices) != len(temporal_stride_probs):
            raise ValueError(
                f"temporal_stride_choices/probs length mismatch: "
                f"{len(temporal_stride_choices)} vs {len(temporal_stride_probs)}"
            )
        if any(x <= 0 for x in temporal_stride_choices):
            raise ValueError(
                f"temporal_stride_choices must be positive, got {temporal_stride_choices}"
            )
        prob_sum = float(temporal_stride_probs.sum())
        if not np.isfinite(prob_sum) or prob_sum <= 0.0:
            raise ValueError(
                f"temporal_stride_probs must have a positive finite sum, got {temporal_stride_probs}"
            )
        temporal_stride_probs = temporal_stride_probs / prob_sum
        if (
            context_sphere_blur_camera != "random"
            and context_sphere_blur_camera not in ROBOTWIN_ALL_CAMERAS
        ):
            raise ValueError(
                f"Unsupported context_sphere_blur_camera: {context_sphere_blur_camera}"
            )
        if float(context_sphere_blur_radius_min) <= 0 or float(context_sphere_blur_radius_max) <= 0:
            raise ValueError("context_sphere_blur radius range must be positive.")
        if float(context_sphere_blur_radius_min) > float(context_sphere_blur_radius_max):
            raise ValueError("context_sphere_blur_radius_min cannot exceed radius_max.")
        if float(context_sphere_blur_sigma_min) < 0 or float(context_sphere_blur_sigma_max) < 0:
            raise ValueError("context_sphere_blur sigma range must be non-negative.")
        if float(context_sphere_blur_sigma_min) > float(context_sphere_blur_sigma_max):
            raise ValueError("context_sphere_blur_sigma_min cannot exceed sigma_max.")
        if (
            float(context_sphere_blur_min_radius_px) <= 0
            or float(context_sphere_blur_max_radius_px) <= 0
        ):
            raise ValueError("context_sphere_blur pixel radius range must be positive.")
        if float(context_sphere_blur_min_radius_px) > float(context_sphere_blur_max_radius_px):
            raise ValueError("context_sphere_blur_min_radius_px cannot exceed max_radius_px.")

        self.base_path = Path(base_path)
        self.num_frames = int(num_frames)
        self.repeat = int(repeat)
        self.split = split
        self.stride = int(stride)
        self.first_round_prob = float(first_round_prob)
        self.context_length = int(context_length)
        self.sample_mode = str(sample_mode)
        self.sparse_history_stride = int(sparse_history_stride)
        self.temporal_stride_choices = temporal_stride_choices
        self.temporal_stride_probs = temporal_stride_probs
        self.context_sphere_blur_prob = float(context_sphere_blur_prob)
        self.context_sphere_blur_camera = str(context_sphere_blur_camera)
        self.context_sphere_blur_radius_min = float(context_sphere_blur_radius_min)
        self.context_sphere_blur_radius_max = float(context_sphere_blur_radius_max)
        self.context_sphere_blur_sigma_min = float(context_sphere_blur_sigma_min)
        self.context_sphere_blur_sigma_max = float(context_sphere_blur_sigma_max)
        self.context_sphere_blur_min_radius_px = float(context_sphere_blur_min_radius_px)
        self.context_sphere_blur_max_radius_px = float(context_sphere_blur_max_radius_px)
        self.context_sphere_blur_position_noise_std = float(context_sphere_blur_position_noise_std)
        self.traj_radius = int(traj_radius)
        self.traj_radius_mode = traj_radius_mode
        self.traj_min_radius = int(traj_min_radius)
        self.traj_max_radius = int(traj_max_radius)
        self.traj_ref_depth = float(traj_ref_depth)
        self.traj_near_depth = float(traj_near_depth)
        self.traj_far_depth = float(traj_far_depth)
        self.output_raymap = bool(output_raymap)
        self.ray_o_vmin = float(ray_o_vmin)
        self.ray_o_vmax = float(ray_o_vmax)
        self.ray_d_vmin = float(ray_d_vmin)
        self.ray_d_vmax = float(ray_d_vmax)
        self.raymap_mode = raymap_mode
        if resize_to is None and view_mode == "single":
            resize_to = (256, 320)
        self.resize_to = resize_to
        self.type = dataset_type
        self.view_mode = view_mode
        self.single_camera = single_camera
        self.endpose_quat_order = endpose_quat_order
        self.prompt = prompt
        self.output_fps = 30.0
        self.load_from_cache = False

        if camera_names is not None:
            self.camera_names = list(camera_names)
        elif view_mode == "single":
            self.camera_names = [single_camera]
        else:
            self.camera_names = list(ROBOTWIN_THREE_VIEW_CAMERAS)
        for cam in self.camera_names:
            if cam not in ROBOTWIN_ALL_CAMERAS:
                raise ValueError(f"Unsupported camera name: {cam}")
        if (
            self.context_sphere_blur_prob > 0.0
            and self.context_sphere_blur_camera != "random"
            and self.context_sphere_blur_camera not in self.camera_names
        ):
            raise ValueError(
                f"context_sphere_blur_camera={self.context_sphere_blur_camera} is not active; "
                f"active cameras={self.camera_names}"
            )

        self.episodes = self._discover_episodes(
            train_episodes_per_task=int(train_episodes_per_task),
            val_episodes_per_task=int(val_episodes_per_task),
            task_limit=task_limit,
        )
        self.episode_info: List[dict] = []
        self.sample_indices: List[Tuple[int, int]] = []
        for ep in self.episodes:
            info = self._build_episode_info(ep)
            if info is None:
                continue
            ep_idx = len(self.episode_info)
            self.episode_info.append(info)
            if self.sample_mode == "fixed_sparse_history":
                horizon = self.num_frames - self.context_length
                min_start = 1
                max_start = info["T"] - horizon + 1
            elif self.sample_mode == "random_temporal_stride":
                max_temporal_stride = max(self.temporal_stride_choices)
                min_start = 0
                max_start = info["T"] - (self.num_frames - 2) * max_temporal_stride
            else:
                min_start = 0
                max_start = info["T"] - self.num_frames + 1
            for start in range(min_start, max_start, self.stride):
                self.sample_indices.append((ep_idx, start))
        self.total_samples = len(self.sample_indices)
        self.length = self.total_samples * self.repeat
        print(
            "[RoboTwinDataset] "
            f"base={self.base_path} split={self.split} episodes={len(self.episode_info)} "
            f"samples={self.total_samples} view_mode={self.view_mode} cameras={self.camera_names} "
            f"fps=30 resize_to={self.resize_to} sample_mode={self.sample_mode} "
            f"sparse_history_stride={self.sparse_history_stride} "
            f"temporal_stride_choices={self.temporal_stride_choices} "
            f"temporal_stride_probs={self.temporal_stride_probs.tolist()} "
            f"context_sphere_blur_prob={self.context_sphere_blur_prob} "
            f"quat_order={self.endpose_quat_order}"
        )

    def _discover_episodes(
        self, train_episodes_per_task: int, val_episodes_per_task: int, task_limit: Optional[int]
    ) -> List[dict]:
        if not self.base_path.exists():
            raise FileNotFoundError(f"RoboTwin dataset root not found: {self.base_path}")
        episodes: List[dict] = []
        task_dirs = [p for p in sorted(self.base_path.iterdir()) if p.is_dir()]
        if task_limit is not None:
            task_dirs = task_dirs[: int(task_limit)]
        for task_dir in task_dirs:
            run_dirs = [
                p for p in sorted(task_dir.iterdir()) if p.is_dir() and (p / "data").is_dir()
            ]
            for run_dir in run_dirs:
                files = sorted((run_dir / "data").glob("*.hdf5"), key=_episode_sort_key)
                if self.split == "train":
                    selected = files[:train_episodes_per_task]
                elif self.split == "val":
                    selected = files[
                        train_episodes_per_task : train_episodes_per_task + val_episodes_per_task
                    ]
                else:
                    selected = files
                for path in selected:
                    ep_id, _ = _episode_sort_key(path)
                    episodes.append(
                        {
                            "task": task_dir.name,
                            "run": run_dir.name,
                            "episode_index": int(ep_id) if isinstance(ep_id, int) else -1,
                            "path": path,
                        }
                    )
        return episodes

    def _build_episode_info(self, ep: dict) -> Optional[dict]:
        try:
            with h5py.File(ep["path"], "r") as f:
                n = int(f["endpose/left_endpose"].shape[0])
                for cam in self.camera_names:
                    n = min(n, int(f[f"observation/{cam}/rgb"].shape[0]))
        except Exception as exc:
            print(f"[RoboTwinDataset] skip {ep['path']}: {exc}")
            return None
        if n < self.num_frames:
            return None
        out = dict(ep)
        out["T"] = n
        return out

    def get_constant_head_ray_maps(self) -> Tuple[List[Image.Image], List[Image.Image]]:
        if "head_camera" not in self.camera_names:
            raise ValueError("head_camera is not active in this dataset instance.")
        if not self.episode_info:
            raise RuntimeError("Dataset has no valid episodes for building head ray maps.")
        info = self.episode_info[0]
        with h5py.File(info["path"], "r") as f:
            frame = _decode_rgb_bytes(f["observation/head_camera/rgb"][0])
            ori_h, ori_w = frame.shape[:2]
            out_h, out_w = self.resize_to if self.resize_to is not None else (ori_h, ori_w)
            K = _scale_intrinsic(
                np.asarray(f["observation/head_camera/intrinsic_cv"][0], dtype=np.float32),
                ori_h,
                ori_w,
                out_h,
                out_w,
            )
            c2w = np.linalg.inv(_to_4x4_w2c(f["observation/head_camera/extrinsic_cv"][0])).astype(
                np.float32
            )
        ro, rd = generate_raymap(K, c2w, out_h, out_w)
        ray_o = Image.fromarray(to_uint8_img(ro, self.ray_o_vmin, self.ray_o_vmax))
        ray_d = Image.fromarray(to_uint8_img(rd, self.ray_d_vmin, self.ray_d_vmax))
        return [ray_o.copy() for _ in range(self.num_frames)], [
            ray_d.copy() for _ in range(self.num_frames)
        ]

    def __len__(self) -> int:
        return self.length

    def _sample_temporal_stride(self) -> int:
        return int(np.random.choice(self.temporal_stride_choices, p=self.temporal_stride_probs))

    def _make_frame_ids(self, start_idx: int) -> np.ndarray:
        if self.context_length > 1:
            if np.random.rand() < self.first_round_prob:
                horizon = self.num_frames - self.context_length
                if self.sample_mode == "random_temporal_stride":
                    temporal_stride = self._sample_temporal_stride()
                    horizon_ids = [temporal_stride * i for i in range(1, horizon + 1)]
                    return np.array([0] * self.context_length + horizon_ids, dtype=np.int64)
                return np.array(
                    [0] * self.context_length + list(range(1, horizon + 1)), dtype=np.int64
                )
            if self.sample_mode == "fixed_sparse_history":
                t = int(start_idx)
                s = int(self.sparse_history_stride)
                horizon = self.num_frames - self.context_length
                history_ids = [
                    0,
                    max(0, t - 1 - 3 * s),
                    max(0, t - 1 - 2 * s),
                    max(0, t - 1 - s),
                    max(0, t - 1),
                ]
                horizon_ids = list(range(t, t + horizon))
                return np.array(history_ids + horizon_ids, dtype=np.int64)
            if self.sample_mode == "random_temporal_stride":
                temporal_stride = self._sample_temporal_stride()
                consecutive_ids = (
                    start_idx + np.arange(self.num_frames - 1, dtype=np.int64) * temporal_stride
                )
                return np.concatenate([[0], consecutive_ids]).astype(np.int64)
            consecutive_ids = np.arange(start_idx, start_idx + self.num_frames - 1, dtype=np.int64)
            return np.concatenate([[0], consecutive_ids]).astype(np.int64)
        return np.arange(start_idx, start_idx + self.num_frames, dtype=np.int64)

    def _read_episode_arrays(self, path: Path, frame_ids: np.ndarray):
        unique_ids = sorted({int(i) for i in frame_ids.tolist()})
        unique_to_pos = {fid: pos for pos, fid in enumerate(unique_ids)}
        with h5py.File(path, "r") as f:
            left_endpose = np.asarray(f["endpose/left_endpose"][unique_ids], dtype=np.float32)
            right_endpose = np.asarray(f["endpose/right_endpose"][unique_ids], dtype=np.float32)
            left_gripper = np.asarray(f["endpose/left_gripper"][unique_ids], dtype=np.float32)
            right_gripper = np.asarray(f["endpose/right_gripper"][unique_ids], dtype=np.float32)
            abs_unique = _make_abs_actions_from_endpose(
                left_endpose,
                right_endpose,
                left_gripper,
                right_gripper,
                self.endpose_quat_order,
            )
            abs_actions = np.stack(
                [abs_unique[unique_to_pos[int(fid)]] for fid in frame_ids], axis=0
            )

            frames_by_cam: Dict[str, List[np.ndarray]] = {}
            w2c_by_cam: Dict[str, np.ndarray] = {}
            K_by_cam: Dict[str, np.ndarray] = {}
            for cam in self.camera_names:
                rgb_ds = f[f"observation/{cam}/rgb"]
                frame_cache = {fid: _decode_rgb_bytes(rgb_ds[fid]) for fid in unique_ids}
                frames_by_cam[cam] = [frame_cache[int(fid)] for fid in frame_ids]
                w2c_unique = np.stack(
                    [_to_4x4_w2c(f[f"observation/{cam}/extrinsic_cv"][fid]) for fid in unique_ids],
                    axis=0,
                )
                w2c_by_cam[cam] = np.stack(
                    [w2c_unique[unique_to_pos[int(fid)]] for fid in frame_ids], axis=0
                )
                K_unique = np.stack(
                    [
                        np.asarray(f[f"observation/{cam}/intrinsic_cv"][fid], dtype=np.float32)
                        for fid in unique_ids
                    ],
                    axis=0,
                )
                K_by_cam[cam] = np.stack(
                    [K_unique[unique_to_pos[int(fid)]] for fid in frame_ids], axis=0
                )
        return abs_actions, frames_by_cam, w2c_by_cam, K_by_cam

    def _sphere_blur_is_visible(
        self,
        blur_info: dict,
        frames_by_cam: Dict[str, List[np.ndarray]],
        w2c_by_cam: Dict[str, np.ndarray],
        K_by_cam: Dict[str, np.ndarray],
    ) -> bool:
        cam = blur_info["camera"]
        first_frame = frames_by_cam[cam][0]
        ori_h, ori_w = first_frame.shape[:2]
        out_h, out_w = self.resize_to if self.resize_to is not None else (ori_h, ori_w)
        for i in range(1, min(self.context_length, len(frames_by_cam[cam]))):
            K = _scale_intrinsic(K_by_cam[cam][i], ori_h, ori_w, out_h, out_w)
            projected = _project_sphere_to_image(
                blur_info["center_world"],
                blur_info["radius_m"],
                w2c_by_cam[cam][i],
                K,
                out_h,
                out_w,
                self.context_sphere_blur_min_radius_px,
                self.context_sphere_blur_max_radius_px,
            )
            if projected is not None:
                return True
        return False

    def _sample_context_sphere_blur(
        self,
        abs_actions: np.ndarray,
        frames_by_cam: Dict[str, List[np.ndarray]],
        w2c_by_cam: Dict[str, np.ndarray],
        K_by_cam: Dict[str, np.ndarray],
    ) -> Optional[dict]:
        if self.context_sphere_blur_prob <= 0.0 or self.context_length <= 1:
            return None
        if np.random.rand() >= self.context_sphere_blur_prob:
            return None
        context_end = min(self.context_length, len(abs_actions))
        candidate_centers = []
        for i in range(1, context_end):
            for start in (0, 8):
                center = _eef_projection_center_world(abs_actions[i], start)
                if np.all(np.isfinite(center)):
                    candidate_centers.append(center)
        if not candidate_centers:
            return None
        active_cameras = list(self.camera_names)
        for _ in range(32):
            camera = (
                str(np.random.choice(active_cameras))
                if self.context_sphere_blur_camera == "random"
                else self.context_sphere_blur_camera
            )
            center = np.array(
                candidate_centers[int(np.random.randint(len(candidate_centers)))], dtype=np.float32
            )
            if self.context_sphere_blur_position_noise_std > 0.0:
                center = center + np.random.normal(
                    0.0,
                    self.context_sphere_blur_position_noise_std,
                    size=3,
                ).astype(np.float32)
            blur_info = {
                "camera": camera,
                "center_world": center,
                "radius_m": float(
                    np.random.uniform(
                        self.context_sphere_blur_radius_min, self.context_sphere_blur_radius_max
                    )
                ),
                "sigma_px": float(
                    np.random.uniform(
                        self.context_sphere_blur_sigma_min, self.context_sphere_blur_sigma_max
                    )
                ),
            }
            if self._sphere_blur_is_visible(blur_info, frames_by_cam, w2c_by_cam, K_by_cam):
                return blur_info
        return None

    def _apply_context_sphere_blur(
        self,
        image: Image.Image,
        blur_info: Optional[dict],
        cam: str,
        frame_pos: int,
        w2c: np.ndarray,
        K: np.ndarray,
    ) -> Image.Image:
        if blur_info is None or cam != blur_info["camera"]:
            return image
        if frame_pos <= 0 or frame_pos >= self.context_length:
            return image
        return _blur_projected_sphere_region(
            image,
            blur_info["center_world"],
            blur_info["radius_m"],
            w2c,
            K,
            blur_info["sigma_px"],
            self.context_sphere_blur_min_radius_px,
            self.context_sphere_blur_max_radius_px,
        )

    def __getitem__(self, idx: int) -> dict:
        if self.total_samples == 0:
            raise IndexError("Dataset has no valid samples")
        sample_idx = idx % self.total_samples
        ep_idx, start_idx = self.sample_indices[sample_idx]
        info = self.episode_info[ep_idx]
        frame_ids = self._make_frame_ids(start_idx)
        abs_actions, frames_by_cam, w2c_by_cam, K_by_cam = self._read_episode_arrays(
            info["path"], frame_ids
        )
        sphere_blur_info = self._sample_context_sphere_blur(
            abs_actions, frames_by_cam, w2c_by_cam, K_by_cam
        )

        cam_frames: Dict[str, List[Image.Image]] = {cam: [] for cam in self.camera_names}
        cam_traj_maps: Dict[str, List[Image.Image]] = {cam: [] for cam in self.camera_names}
        cam_ray_o: Dict[str, List[Image.Image]] = {cam: [] for cam in self.camera_names}
        cam_ray_d: Dict[str, List[Image.Image]] = {cam: [] for cam in self.camera_names}

        for cam in self.camera_names:
            first_frame = frames_by_cam[cam][0]
            ori_h, ori_w = first_frame.shape[:2]
            out_h, out_w = self.resize_to if self.resize_to is not None else (ori_h, ori_w)
            traj_maps = []
            for i in range(len(frame_ids)):
                K = _scale_intrinsic(K_by_cam[cam][i], ori_h, ori_w, out_h, out_w)
                traj_maps.append(
                    generate_traj_map(
                        abs_actions[i : i + 1],
                        w2c_by_cam[cam][i : i + 1],
                        K,
                        out_h,
                        out_w,
                        radius=self.traj_radius,
                        radius_mode=self.traj_radius_mode,
                        min_radius=self.traj_min_radius,
                        max_radius=self.traj_max_radius,
                        ref_depth=self.traj_ref_depth,
                        near_depth=self.traj_near_depth,
                        far_depth=self.traj_far_depth,
                    )[0]
                )
            for i, fid in enumerate(frame_ids):
                img = Image.fromarray(frames_by_cam[cam][i])
                if self.resize_to is not None:
                    img = img.resize((out_w, out_h), Image.BILINEAR)
                K_for_blur = _scale_intrinsic(K_by_cam[cam][i], ori_h, ori_w, out_h, out_w)
                img = self._apply_context_sphere_blur(
                    img,
                    sphere_blur_info,
                    cam,
                    i,
                    w2c_by_cam[cam][i],
                    K_for_blur,
                )
                cam_frames[cam].append(img)
                cam_traj_maps[cam].append(Image.fromarray(traj_maps[i]))
                if self.output_raymap:
                    K = _scale_intrinsic(K_by_cam[cam][i], ori_h, ori_w, out_h, out_w)
                    c2w = np.linalg.inv(w2c_by_cam[cam][i]).astype(np.float32)
                    ro, rd = generate_raymap(K, c2w, out_h, out_w)
                    cam_ray_o[cam].append(
                        Image.fromarray(to_uint8_img(ro, self.ray_o_vmin, self.ray_o_vmax))
                    )
                    cam_ray_d[cam].append(
                        Image.fromarray(to_uint8_img(rd, self.ray_d_vmin, self.ray_d_vmax))
                    )

        video_list, control_list, ray_o_list, ray_d_list = [], [], [], []
        for i in range(len(frame_ids)):
            video_list.append(
                Image.fromarray(
                    np.concatenate(
                        [np.asarray(cam_frames[cam][i]) for cam in self.camera_names], axis=0
                    )
                )
            )
            control_list.append(
                Image.fromarray(
                    np.concatenate(
                        [np.asarray(cam_traj_maps[cam][i]) for cam in self.camera_names], axis=0
                    )
                )
            )
            if self.output_raymap:
                ray_o_list.append(
                    Image.fromarray(
                        np.concatenate(
                            [np.asarray(cam_ray_o[cam][i]) for cam in self.camera_names], axis=0
                        )
                    )
                )
                ray_d_list.append(
                    Image.fromarray(
                        np.concatenate(
                            [np.asarray(cam_ray_d[cam][i]) for cam in self.camera_names], axis=0
                        )
                    )
                )

        if self.type == "vace":
            sample = {
                "video": video_list,
                "vace_reference_image": [video_list[0]],
                "vace_video": control_list,
                "prompt": self.prompt,
            }
        else:
            sample = {
                "video": video_list,
                "reference_image": [video_list[0]],
                "control_video": control_list,
                "prompt": self.prompt,
            }
        if self.output_raymap:
            sample["ray_map_o"] = ray_o_list
            sample["ray_map_d"] = ray_d_list
        sphere_blur_meta = None
        if sphere_blur_info is not None:
            sphere_blur_meta = {
                "camera": sphere_blur_info["camera"],
                "center_world": np.asarray(sphere_blur_info["center_world"], dtype=float).tolist(),
                "radius_m": float(sphere_blur_info["radius_m"]),
                "sigma_px": float(sphere_blur_info["sigma_px"]),
                "context_frame_positions": list(range(1, min(self.context_length, len(frame_ids)))),
            }
        sample["meta"] = {
            "task": info["task"],
            "run": info["run"],
            "episode_index": info["episode_index"],
            "path": str(info["path"]),
            "frame_ids": frame_ids.tolist(),
            "camera_names": list(self.camera_names),
            "view_mode": self.view_mode,
            "traj_source": "robotwin_endpose",
            "raymap_frame": "robotwin_world_cv_extrinsic",
            "context_sphere_blur": sphere_blur_meta,
        }
        return sample


def _as_rgb_array(image_like) -> np.ndarray:
    return np.asarray(
        image_like.convert("RGB") if isinstance(image_like, Image.Image) else image_like,
        dtype=np.uint8,
    )


def make_debug_grid_video(sample: dict, output_path: Path, fps: float = 30.0) -> None:
    frames = []
    for i in range(len(sample["video"])):
        cols = [
            _as_rgb_array(sample["video"][i]),
            _as_rgb_array(sample["vace_video"][i]),
            _as_rgb_array(sample["ray_map_o"][i]),
            _as_rgb_array(sample["ray_map_d"][i]),
        ]
        frames.append(np.concatenate(cols, axis=1))
    write_video_rgb(frames, Path(output_path), fps=fps)
