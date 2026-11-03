#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from scipy.spatial.transform import Rotation

try:
    import yaml
except ImportError:
    yaml = None

from .projection import (
    generate_raymap,
    generate_traj_map,
    ColorListLeft,
    ColorListRight,
    EndEffectorPts,
    Gripper2EEFCvt,
    compute_depth_radius,
    get_transformation_matrix_from_quat,
    grip_to_left_color,
    grip_to_right_color,
    to_uint8_img,
    write_video_rgb,
)

ROBOTWIN_THREE_VIEW_CAMERAS = ("head_camera", "left_camera", "right_camera")
ROBOTWIN_ALL_CAMERAS = ("front_camera", "head_camera", "left_camera", "right_camera")
DEFAULT_FK_CONFIG_PATH = "../wovr-assets/robot/aloha-agilex/config.yml"


def _draw_alpha_ellipse(
    base: np.ndarray, center_xy: np.ndarray, radius: int, color: Sequence[int], alpha: int
) -> np.ndarray:
    mask = Image.new("L", (base.shape[1], base.shape[0]), 0)
    draw = ImageDraw.Draw(mask)
    x, y = int(center_xy[0]), int(center_xy[1])
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=int(np.clip(alpha, 0, 255)))
    fg = Image.new("RGB", (base.shape[1], base.shape[0]), tuple(int(c) for c in color))
    return np.asarray(Image.composite(fg, Image.fromarray(base), mask), dtype=np.uint8)


def generate_traj_map_alpha(
    abs_actions: np.ndarray,
    w2c: np.ndarray,
    intrinsic: np.ndarray,
    h: int,
    w: int,
    radius: int = 40,
    radius_mode: str = "perspective",
    min_radius: int = 14,
    max_radius: int = 58,
    ref_depth: float = 0.30,
    near_depth: float = 0.19,
    far_depth: float = 0.69,
    circle_alpha: int = 160,
    local_x_offset: float = 0.2,
    axis_width: int = 4,
) -> List[np.ndarray]:
    """Projection control map with alpha-blended endpoint circles.

    This keeps the same projection, radius, gripper color, and axis-line logic
    as generate_traj_map; only the filled left/right endpoint circles are alpha
    composited so overlapping endpoints do not fully hide each other.
    """
    abs_actions = np.asarray(abs_actions, dtype=np.float32)
    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    w2c = np.asarray(w2c, dtype=np.float32)
    if w2c.ndim == 2:
        w2c = np.repeat(w2c[None], abs_actions.shape[0], axis=0)

    ee_key_pts = EndEffectorPts.reshape(1, 4, 4).transpose(0, 2, 1)
    cvt_matrix = np.eye(4, dtype=np.float32).reshape(1, 4, 4)
    cvt_matrix[:, 0, 3] = float(local_x_offset)
    pose_l_mat = get_transformation_matrix_from_quat(abs_actions[:, 0:7])
    pose_r_mat = get_transformation_matrix_from_quat(abs_actions[:, 8:15])
    ee2cam_l = np.matmul(np.matmul(w2c, pose_l_mat), cvt_matrix)
    ee2cam_r = np.matmul(np.matmul(w2c, pose_r_mat), cvt_matrix)
    pts_l = np.matmul(ee2cam_l, ee_key_pts)
    pts_r = np.matmul(ee2cam_r, ee_key_pts)

    intr = intrinsic.reshape(1, 3, 3)
    uvs_l = np.matmul(intr, pts_l[:, :3, :])
    uvs_l = (uvs_l / (pts_l[:, 2:3, :] + 1e-8))[:, :2, :].transpose(0, 2, 1).astype(np.int64)
    uvs_r = np.matmul(intr, pts_r[:, :3, :])
    uvs_r = (uvs_r / (pts_r[:, 2:3, :] + 1e-8))[:, :2, :].transpose(0, 2, 1).astype(np.int64)

    img_list: List[np.ndarray] = []
    for i in range(abs_actions.shape[0]):
        img = np.zeros((h, w, 3), dtype=np.uint8) + 50
        color_l = grip_to_left_color(abs_actions[i, 7].item() / 120.0)
        color_r = grip_to_right_color(abs_actions[i, 15].item() / 120.0)

        for points, pts, color in zip(
            [uvs_l[i], uvs_r[i]], [pts_l[i], pts_r[i]], [color_l, color_r]
        ):
            if float(pts[2, 0]) <= 1e-6:
                continue
            base = np.asarray(points[0])
            if base[0] < 0 or base[0] >= w or base[1] < 0 or base[1] >= h:
                continue
            radius_i = compute_depth_radius(
                float(pts[2, 0]),
                radius,
                radius_mode,
                min_radius,
                max_radius,
                ref_depth,
                near_depth,
                far_depth,
            )
            img = _draw_alpha_ellipse(img, base, radius_i, color, circle_alpha)

        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        for points, pts, colors in zip(
            [uvs_l[i], uvs_r[i]], [pts_l[i], pts_r[i]], [ColorListLeft, ColorListRight]
        ):
            if float(pts[2, 0]) <= 1e-6:
                continue
            base = np.asarray(points[0])
            if base[0] < 0 or base[0] >= w or base[1] < 0 or base[1] >= h:
                continue
            for point_idx, point in enumerate(points):
                if point_idx == 0 or float(pts[2, point_idx]) <= 1e-6:
                    continue
                draw.line(
                    [tuple(map(int, base)), tuple(map(int, point[:2]))],
                    fill=tuple(colors[point_idx - 1]),
                    width=int(axis_width),
                )
        img_list.append(np.asarray(pil, dtype=np.uint8))
    return img_list


def _episode_sort_key(path: Path) -> Tuple[int, str]:
    match = re.search(r"episode(\d+)", path.stem)
    if match:
        return int(match.group(1)), path.name
    return 10**9, path.name


def _decode_rgb_bytes(raw, source_color_order: str = "rgb") -> np.ndarray:
    if isinstance(raw, np.bytes_):
        raw = raw.tobytes()
    elif isinstance(raw, np.ndarray):
        raw = raw.tobytes()
    elif not isinstance(raw, (bytes, bytearray)):
        raw = bytes(raw)
    image = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)
    if source_color_order == "bgr":
        image = image[..., ::-1].copy()
    elif source_color_order != "rgb":
        raise ValueError(f"Unsupported source_color_order: {source_color_order}")
    return image


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


def _parse_base_paths(base_path: str) -> List[Path]:
    if isinstance(base_path, (list, tuple)):
        if not base_path:
            raise ValueError("base_path is empty")
        return [Path(item) for item in base_path]
    text = str(base_path).strip()
    if not text:
        raise ValueError("base_path is empty")
    if text.startswith("["):
        items = json.loads(text)
    else:
        items = [item.strip() for item in text.split(",") if item.strip()]
    if not items:
        raise ValueError(f"No valid base paths parsed from {base_path!r}")
    return [Path(item) for item in items]


def _find_action_source(f: h5py.File) -> Tuple[str, str]:
    for key in ("actions", "action"):
        if key not in f:
            continue
        obj = f[key]
        if isinstance(obj, h5py.Dataset):
            if len(obj.shape) == 2 and obj.shape[1] >= 14:
                return "vector", key
            continue
        if isinstance(obj, h5py.Group):
            vector_path = f"{key}/vector"
            if "vector" in obj and len(f[vector_path].shape) == 2 and f[vector_path].shape[1] >= 14:
                return "vector", vector_path
            fields = ("left_arm", "left_gripper", "right_arm", "right_gripper")
            if all(field in obj for field in fields):
                return "group", key
    if "joint_action" in f:
        obj = f["joint_action"]
        fields = ("left_arm", "left_gripper", "right_arm", "right_gripper")
        if isinstance(obj, h5py.Group) and all(field in obj for field in fields):
            return "group", "joint_action"
    raise KeyError("Cannot find actions/action/joint_action fields in RoboTwin hdf5 file.")


def _action_source_length(f: h5py.File) -> int:
    kind, source = _find_action_source(f)
    if kind == "vector":
        return int(f[source].shape[0])
    return int(f[f"{source}/left_arm"].shape[0])


def _read_action_components(
    f: h5py.File,
    indices: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    kind, source = _find_action_source(f)
    if kind == "vector":
        action = np.asarray(f[source][indices], dtype=np.float32)
        if action.ndim != 2 or action.shape[1] < 14:
            raise ValueError(f"Expected {source} to have shape (T, >=14), got {action.shape}")
        return action[:, 0:6], action[:, 6], action[:, 7:13], action[:, 13]
    return (
        np.asarray(f[f"{source}/left_arm"][indices], dtype=np.float32),
        np.asarray(f[f"{source}/left_gripper"][indices], dtype=np.float32),
        np.asarray(f[f"{source}/right_arm"][indices], dtype=np.float32),
        np.asarray(f[f"{source}/right_gripper"][indices], dtype=np.float32),
    )


def _load_yaml(path: Path) -> dict:
    if yaml is None:
        raise ImportError("PyYAML is required for RoboTwin action-FK dataset.")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _gripper_joint_names(config: dict) -> Tuple[str, str]:
    if "gripper_joint_names" in config:
        return tuple(config["gripper_joint_names"])
    return config["gripper_name"][0]["base"], config["gripper_name"][1]["base"]


def _transformed_eef_pose(config: dict, joint_global_transform: np.ndarray) -> np.ndarray:
    global_trans_matrix = np.asarray(config["global_trans_matrix"], dtype=np.float64)
    delta_matrix = np.asarray(config["delta_matrix"], dtype=np.float64)
    dis = float(config.get("gripper_bias", 0.0)) - 0.12
    rot = (
        np.asarray(joint_global_transform[:3, :3], dtype=np.float64)
        @ global_trans_matrix
        @ delta_matrix
    )
    pos = np.asarray(joint_global_transform[:3, 3], dtype=np.float64) + rot @ np.asarray(
        [dis, 0.0, 0.0]
    )
    quat_xyzw = Rotation.from_matrix(rot).as_quat()
    quat_wxyz = np.asarray(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64
    )
    return np.concatenate([pos, quat_wxyz]).astype(np.float64)


def _make_abs_actions_from_fk_wxyz(
    left_pose_wxyz: np.ndarray,
    right_pose_wxyz: np.ndarray,
    left_gripper: np.ndarray,
    right_gripper: np.ndarray,
) -> np.ndarray:
    left_pose_wxyz = np.asarray(left_pose_wxyz, dtype=np.float32)
    right_pose_wxyz = np.asarray(right_pose_wxyz, dtype=np.float32)
    out = np.zeros((left_pose_wxyz.shape[0], 16), dtype=np.float32)
    out[:, 0:3] = left_pose_wxyz[:, 0:3]
    out[:, 3:7] = left_pose_wxyz[:, [4, 5, 6, 3]]
    out[:, 7] = np.clip(np.asarray(left_gripper, dtype=np.float32), 0.0, 1.0) * 120.0
    out[:, 8:11] = right_pose_wxyz[:, 0:3]
    out[:, 11:15] = right_pose_wxyz[:, [4, 5, 6, 3]]
    out[:, 15] = np.clip(np.asarray(right_gripper, dtype=np.float32), 0.0, 1.0) * 120.0
    return out


def _parse_xyz(text: Optional[str], default=(0.0, 0.0, 0.0)) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(x) for x in text.split()], dtype=np.float64)


def _make_transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    out[:3, 3] = xyz
    return out


def _joint_motion(joint_type: str, axis: np.ndarray, value: float) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    if joint_type in {"revolute", "continuous"}:
        norm = float(np.linalg.norm(axis))
        if norm > 1e-12:
            out[:3, :3] = Rotation.from_rotvec(axis / norm * float(value)).as_matrix()
    elif joint_type == "prismatic":
        out[:3, 3] = axis * float(value)
    return out


class _UrdfSerialFK:
    def __init__(self, urdf_path: Path, root_pose: Sequence[float]):
        self.children_by_parent: Dict[str, List[dict]] = {}
        self.child_links = set()
        robot = ET.parse(urdf_path).getroot()
        for elem in robot.findall("joint"):
            name = elem.attrib["name"]
            joint_type = elem.attrib.get("type", "fixed")
            parent = elem.find("parent").attrib["link"]
            child = elem.find("child").attrib["link"]
            origin = elem.find("origin")
            axis = elem.find("axis")
            joint = {
                "name": name,
                "type": joint_type,
                "parent": parent,
                "child": child,
                "origin": _make_transform(
                    _parse_xyz(origin.attrib.get("xyz") if origin is not None else None),
                    _parse_xyz(origin.attrib.get("rpy") if origin is not None else None),
                ),
                "axis": _parse_xyz(
                    axis.attrib.get("xyz") if axis is not None else None, default=(0.0, 0.0, 1.0)
                ),
            }
            self.children_by_parent.setdefault(parent, []).append(joint)
            self.child_links.add(child)

        roots = sorted(set(self.children_by_parent) - self.child_links)
        if not roots:
            raise ValueError(f"Cannot infer URDF root link from {urdf_path}")
        self.root_link = roots[0]

        root_pose = np.asarray(root_pose, dtype=np.float64)
        self.root_transform = np.eye(4, dtype=np.float64)
        self.root_transform[:3, 3] = root_pose[:3]
        quat_wxyz = root_pose[3:7]
        self.root_transform[:3, :3] = Rotation.from_quat(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        ).as_matrix()

    def forward(self, joint_values: Dict[str, float]) -> Dict[str, np.ndarray]:
        link_transforms = {self.root_link: self.root_transform}
        joint_transforms: Dict[str, np.ndarray] = {}
        stack = [self.root_link]
        while stack:
            parent = stack.pop()
            parent_tf = link_transforms[parent]
            for joint in self.children_by_parent.get(parent, []):
                value = float(joint_values.get(joint["name"], 0.0))
                child_tf = (
                    parent_tf @ joint["origin"] @ _joint_motion(joint["type"], joint["axis"], value)
                )
                joint_transforms[joint["name"]] = child_tf
                link_transforms[joint["child"]] = child_tf
                stack.append(joint["child"])
        return joint_transforms


class RoboTwinActionFKDatasetWancontrolmultiview(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path: str = "../wovr-assets/datasets/adjust_bottle",
        num_frames: int = 13,
        repeat: int = 1,
        split: str = "train",
        train_episodes_per_task: int = 40,
        val_episodes_per_task: int = 10,
        val_tail_episodes_per_task: int = 0,
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
        traj_circle_alpha: int = 160,
        traj_local_x_offset: float = 0.2,
        traj_axis_width: int = 4,
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
        fk_config_path: str = DEFAULT_FK_CONFIG_PATH,
        trajectory_source: str = "action_fk",
        source_color_order: str = "rgb",
    ):
        super().__init__()
        if split not in {"train", "val", "all"}:
            raise ValueError(f"Unsupported split: {split}")
        if dataset_type not in {"vace", "control"}:
            raise ValueError(f"Unsupported dataset_type: {dataset_type}")
        if raymap_mode not in ("image", "latent"):
            raise ValueError(f"Unsupported raymap_mode: {raymap_mode}")
        if view_mode not in {"three", "single"}:
            raise ValueError(f"Unsupported view_mode: {view_mode}")
        if single_camera not in ROBOTWIN_ALL_CAMERAS:
            raise ValueError(f"Unsupported single_camera: {single_camera}")
        if endpose_quat_order not in {"wxyz", "xyzw"}:
            raise ValueError(f"Unsupported endpose_quat_order: {endpose_quat_order}")
        if trajectory_source not in {"action_fk", "endpose"}:
            raise ValueError(f"Unsupported trajectory_source: {trajectory_source}")
        if source_color_order not in {"rgb", "bgr"}:
            raise ValueError(f"Unsupported source_color_order: {source_color_order}")
        if not 0 <= int(traj_circle_alpha) <= 255:
            raise ValueError(f"traj_circle_alpha must be in [0, 255], got {traj_circle_alpha}")
        if int(traj_axis_width) <= 0:
            raise ValueError(f"traj_axis_width must be positive, got {traj_axis_width}")
        if sample_mode not in {
            "contiguous",
            "fixed_sparse_history",
            "random_temporal_stride",
            "fixed_sparse_history_random_temporal_stride",
        }:
            raise ValueError(f"Unsupported sample_mode: {sample_mode}")
        if (
            sample_mode in {"fixed_sparse_history", "fixed_sparse_history_random_temporal_stride"}
            and int(context_length) != 5
        ):
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

        self.base_paths = _parse_base_paths(base_path)
        self.base_path = self.base_paths[0]
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
        self.traj_circle_alpha = int(traj_circle_alpha)
        self.traj_local_x_offset = float(traj_local_x_offset)
        self.traj_axis_width = int(traj_axis_width)
        self.output_raymap = bool(output_raymap)
        self.ray_o_vmin = float(ray_o_vmin)
        self.ray_o_vmax = float(ray_o_vmax)
        self.ray_d_vmin = float(ray_d_vmin)
        self.ray_d_vmax = float(ray_d_vmax)
        self.raymap_mode = raymap_mode
        self.vae_spatial_downsample = 16
        self.vae_temporal_downsample = 4
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
        self.fk_config_path = Path(fk_config_path)
        self.trajectory_source = str(trajectory_source)
        self.source_color_order = str(source_color_order)
        self._fk_config = None
        self._fk_engine = None
        self._fk_scene = None
        self._fk_robot = None
        self._fk_name_to_idx = None
        self._fk_left_ee = None
        self._fk_right_ee = None

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
            val_tail_episodes_per_task=int(val_tail_episodes_per_task),
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
            elif self.sample_mode == "fixed_sparse_history_random_temporal_stride":
                horizon = self.num_frames - self.context_length
                max_temporal_stride = max(self.temporal_stride_choices)
                min_start = 1
                max_start = info["T"] - (horizon - 1) * max_temporal_stride
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
            "[RoboTwinActionFKDataset] "
            f"bases={len(self.base_paths)} first_base={self.base_path} split={self.split} episodes={len(self.episode_info)} "
            f"samples={self.total_samples} view_mode={self.view_mode} cameras={self.camera_names} "
            f"fps=30 resize_to={self.resize_to} sample_mode={self.sample_mode} "
            f"sparse_history_stride={self.sparse_history_stride} "
            f"temporal_stride_choices={self.temporal_stride_choices} "
            f"temporal_stride_probs={self.temporal_stride_probs.tolist()} "
            f"context_sphere_blur_prob={self.context_sphere_blur_prob} "
            f"traj_source={self.trajectory_source} "
            f"source_color_order={self.source_color_order} "
            f"traj_alpha={self.traj_circle_alpha} traj_local_x_offset={self.traj_local_x_offset} "
            f"traj_axis_width={self.traj_axis_width} "
            f"fk_config={self.fk_config_path if self.trajectory_source == 'action_fk' else 'unused'}"
        )

    def _discover_episodes(
        self,
        train_episodes_per_task: int,
        val_episodes_per_task: int,
        val_tail_episodes_per_task: int,
        task_limit: Optional[int],
    ) -> List[dict]:
        episodes: List[dict] = []
        roots = self.base_paths[: int(task_limit)] if task_limit is not None else self.base_paths
        for root in roots:
            if not root.exists():
                raise FileNotFoundError(f"RoboTwin dataset root not found: {root}")
            data_dirs: List[Path] = []
            if root.is_dir() and any(root.glob("*.hdf5")):
                data_dirs.append(root)
            elif root.is_dir() and (root / "data").is_dir():
                data_dirs.append(root / "data")
            else:
                task_dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
                for task_dir in task_dirs:
                    run_dirs = [
                        p
                        for p in sorted(task_dir.iterdir())
                        if p.is_dir() and (p / "data").is_dir()
                    ]
                    data_dirs.extend([run_dir / "data" for run_dir in run_dirs])

            for data_dir in data_dirs:
                files = sorted(data_dir.glob("*.hdf5"), key=_episode_sort_key)
                val_tail = max(0, int(val_tail_episodes_per_task))
                if val_tail > 0 and self.split in {"train", "val"}:
                    val_tail = min(val_tail, len(files))
                    if self.split == "train":
                        selected = files[:-val_tail]
                    else:
                        selected = files[-val_tail:]
                elif self.split == "train":
                    selected = files[:train_episodes_per_task]
                elif self.split == "val":
                    selected = files[
                        train_episodes_per_task : train_episodes_per_task + val_episodes_per_task
                    ]
                else:
                    selected = files
                if data_dir.name == "data":
                    run_name = data_dir.parent.name
                    task_name = data_dir.parent.parent.name
                else:
                    run_name = "direct"
                    task_name = data_dir.name
                for path in selected:
                    ep_id, _ = _episode_sort_key(path)
                    episodes.append(
                        {
                            "task": task_name,
                            "run": run_name,
                            "episode_index": int(ep_id) if isinstance(ep_id, int) else -1,
                            "path": path,
                        }
                    )
        return episodes

    def _build_episode_info(self, ep: dict) -> Optional[dict]:
        try:
            with h5py.File(ep["path"], "r") as f:
                if self.trajectory_source == "endpose":
                    required = (
                        "endpose/left_endpose",
                        "endpose/right_endpose",
                        "endpose/left_gripper",
                        "endpose/right_gripper",
                    )
                    n = min(int(f[key].shape[0]) for key in required)
                else:
                    n = _action_source_length(f)
                for cam in self.camera_names:
                    n = min(n, int(f[f"observation/{cam}/rgb"].shape[0]))
        except Exception as exc:
            print(f"[RoboTwinActionFKDataset] skip {ep['path']}: {exc}")
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
            frame = _decode_rgb_bytes(
                f["observation/head_camera/rgb"][0],
                source_color_order=self.source_color_order,
            )
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
                if self.sample_mode in {
                    "random_temporal_stride",
                    "fixed_sparse_history_random_temporal_stride",
                }:
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
            if self.sample_mode == "fixed_sparse_history_random_temporal_stride":
                t = int(start_idx)
                s = int(self.sparse_history_stride)
                temporal_stride = self._sample_temporal_stride()
                horizon = self.num_frames - self.context_length
                history_ids = [
                    0,
                    max(0, t - (3 * s + 1) * temporal_stride),
                    max(0, t - (2 * s + 1) * temporal_stride),
                    max(0, t - (s + 1) * temporal_stride),
                    max(0, t - temporal_stride),
                ]
                horizon_ids = [t + i * temporal_stride for i in range(horizon)]
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

    def _latent_frame_groups(self, num_frames: int) -> List[List[int]]:
        latent_t = (num_frames - 1) // self.vae_temporal_downsample + 1
        groups = [[0] * self.vae_temporal_downsample]
        for i in range(1, latent_t):
            start = 1 + (i - 1) * self.vae_temporal_downsample
            groups.append(
                [min(start + j, num_frames - 1) for j in range(self.vae_temporal_downsample)]
            )
        return groups

    def _generate_raymap_latents_for_camera(
        self,
        w2c_seq: np.ndarray,
        K_seq: np.ndarray,
        image_height: int,
        image_width: int,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if (
            image_height % self.vae_spatial_downsample != 0
            or image_width % self.vae_spatial_downsample != 0
        ):
            raise ValueError(
                f"raymap latent size requires image size divisible by {self.vae_spatial_downsample}, "
                f"got {(image_height, image_width)}"
            )
        latent_h = image_height // self.vae_spatial_downsample
        latent_w = image_width // self.vae_spatial_downsample
        ray_o_list, ray_d_list = [], []
        for group in self._latent_frame_groups(len(w2c_seq)):
            ro_group, rd_group = [], []
            for pos in group:
                K = np.asarray(K_seq[pos], dtype=np.float32).copy()
                K[0, 0] *= latent_w / float(image_width)
                K[0, 2] *= latent_w / float(image_width)
                K[1, 1] *= latent_h / float(image_height)
                K[1, 2] *= latent_h / float(image_height)
                c2w = np.linalg.inv(w2c_seq[pos]).astype(np.float32)
                ro, rd = generate_raymap(K, c2w, latent_h, latent_w)
                ro_group.append(ro)
                rd_group.append(rd)
            ray_o_list.append(np.concatenate(ro_group, axis=-1))
            ray_d_list.append(np.concatenate(rd_group, axis=-1))
        return ray_o_list, ray_d_list

    def _ensure_fk(self) -> None:
        if self._fk_robot is not None:
            return
        if not self.fk_config_path.exists():
            raise FileNotFoundError(f"RoboTwin FK config not found: {self.fk_config_path}")
        self._fk_config = _load_yaml(self.fk_config_path)
        urdf_path = Path(self._fk_config["urdf_path"])
        if not urdf_path.is_absolute():
            urdf_path = self.fk_config_path.parent / urdf_path
        root_pose = self._fk_config.get("robot_pose", [[0, 0, 0, 1, 0, 0, 0]])[0]
        self._fk_robot = _UrdfSerialFK(urdf_path, root_pose)

    def _fk_abs_actions_from_joint_action(
        self,
        left_arm: np.ndarray,
        left_gripper: np.ndarray,
        right_arm: np.ndarray,
        right_gripper: np.ndarray,
    ) -> np.ndarray:
        self._ensure_fk()
        n = min(len(left_arm), len(left_gripper), len(right_arm), len(right_gripper))
        left_pose = np.zeros((n, 7), dtype=np.float64)
        right_pose = np.zeros((n, 7), dtype=np.float64)
        left_joint_names, right_joint_names = self._fk_config["arm_joints_name"]
        left_ee_name, right_ee_name = self._fk_config["ee_joints"]
        for i in range(n):
            joint_values = {}
            for name, value in zip(left_joint_names, np.asarray(left_arm[i], dtype=np.float64)):
                joint_values[name] = float(value)
            for name, value in zip(right_joint_names, np.asarray(right_arm[i], dtype=np.float64)):
                joint_values[name] = float(value)
            joint_transforms = self._fk_robot.forward(joint_values)
            left_pose[i] = _transformed_eef_pose(self._fk_config, joint_transforms[left_ee_name])
            right_pose[i] = _transformed_eef_pose(self._fk_config, joint_transforms[right_ee_name])
        return _make_abs_actions_from_fk_wxyz(
            left_pose, right_pose, left_gripper[:n], right_gripper[:n]
        )

    def _read_episode_arrays(self, path: Path, frame_ids: np.ndarray):
        unique_ids = sorted({int(i) for i in frame_ids.tolist()})
        unique_to_pos = {fid: pos for pos, fid in enumerate(unique_ids)}
        with h5py.File(path, "r") as f:
            if self.trajectory_source == "endpose":
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
            else:
                left_arm, left_gripper, right_arm, right_gripper = _read_action_components(
                    f, unique_ids
                )
                abs_unique = self._fk_abs_actions_from_joint_action(
                    left_arm,
                    left_gripper,
                    right_arm,
                    right_gripper,
                )
            abs_actions = np.stack(
                [abs_unique[unique_to_pos[int(fid)]] for fid in frame_ids], axis=0
            )

            frames_by_cam: Dict[str, List[np.ndarray]] = {}
            w2c_by_cam: Dict[str, np.ndarray] = {}
            K_by_cam: Dict[str, np.ndarray] = {}
            for cam in self.camera_names:
                rgb_ds = f[f"observation/{cam}/rgb"]
                frame_cache = {
                    fid: _decode_rgb_bytes(rgb_ds[fid], source_color_order=self.source_color_order)
                    for fid in unique_ids
                }
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
                # 左臂和右臂的夹爪位置都可以作为模糊中心的候选
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
            # 选择被模糊的视角和模糊中心，模糊中心在当前视角下的投影半径满足要求，并且至少在一个历史帧中可见
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
                    generate_traj_map_alpha(
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
                        circle_alpha=self.traj_circle_alpha,
                        local_x_offset=self.traj_local_x_offset,
                        axis_width=self.traj_axis_width,
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
                if self.output_raymap and self.raymap_mode == "image":
                    K = _scale_intrinsic(K_by_cam[cam][i], ori_h, ori_w, out_h, out_w)
                    c2w = np.linalg.inv(w2c_by_cam[cam][i]).astype(np.float32)
                    ro, rd = generate_raymap(K, c2w, out_h, out_w)
                    cam_ray_o[cam].append(
                        Image.fromarray(to_uint8_img(ro, self.ray_o_vmin, self.ray_o_vmax))
                    )
                    cam_ray_d[cam].append(
                        Image.fromarray(to_uint8_img(rd, self.ray_d_vmin, self.ray_d_vmax))
                    )

            if self.output_raymap and self.raymap_mode == "latent":
                K_scaled_seq = np.stack(
                    [
                        _scale_intrinsic(K_by_cam[cam][i], ori_h, ori_w, out_h, out_w)
                        for i in range(len(frame_ids))
                    ],
                    axis=0,
                )
                ro_lat, rd_lat = self._generate_raymap_latents_for_camera(
                    w2c_by_cam[cam], K_scaled_seq, out_h, out_w
                )
                cam_ray_o[cam].extend(ro_lat)
                cam_ray_d[cam].extend(rd_lat)

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
            if self.output_raymap and self.raymap_mode == "image":
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

        if self.output_raymap and self.raymap_mode == "latent":
            ro_latents, rd_latents = [], []
            for i in range(len(cam_ray_o[self.camera_names[0]])):
                ro_latents.append(
                    np.concatenate([cam_ray_o[cam][i] for cam in self.camera_names], axis=0)
                )
                rd_latents.append(
                    np.concatenate([cam_ray_d[cam][i] for cam in self.camera_names], axis=0)
                )
            ray_o_list = torch.from_numpy(np.stack(ro_latents, axis=0)).permute(3, 0, 1, 2).float()
            ray_d_list = torch.from_numpy(np.stack(rd_latents, axis=0)).permute(3, 0, 1, 2).float()

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
            "traj_source": f"robotwin_{self.trajectory_source}",
            "source_color_order": self.source_color_order,
            "traj_circle_alpha": self.traj_circle_alpha,
            "traj_local_x_offset": self.traj_local_x_offset,
            "traj_axis_width": self.traj_axis_width,
            "fk_config_path": (
                str(self.fk_config_path) if self.trajectory_source == "action_fk" else None
            ),
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
