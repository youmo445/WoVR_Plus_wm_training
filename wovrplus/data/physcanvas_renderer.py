"""RoboTwin FK mesh-motion canvas, adapted to PhysCanvas's gray projection.

The reference renders link6/7/8 meshes on gray (190), with gray (55) grippers,
using FK, per-frame camera extrinsics and pinhole projection. Here the robot
geometry/calibration comes exclusively from RoboTwin's own URDF and HDF5.
No imports, writes or runtime dependencies target the read-only reference repo.
"""
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import h5py
import numpy as np
from PIL import Image

from .robotwin import (
    _UrdfSerialFK, _load_yaml, _make_transform, _parse_xyz,
    _read_action_components, _to_4x4_w2c, _scale_intrinsic, _decode_rgb_bytes,
)


def load_dae_triangles(path: Path) -> np.ndarray:
    """Read the actual URDF COLLADA triangles, rejecting unsupported transforms."""
    root = ET.parse(path).getroot()
    ns = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
    unit = root.find("c:asset/c:unit", ns)
    if unit is None or float(unit.get("meter")) != 1:
        raise ValueError(f"Expected metre-scale RoboTwin mesh: {path}")
    for matrix in root.findall(".//c:node/c:matrix", ns):
        if not np.allclose(np.fromstring(matrix.text, sep=" ").reshape(4, 4), np.eye(4)):
            raise ValueError(f"Nonidentity mesh scene transform: {path}")
    triangles = []
    for mesh in root.findall(".//c:geometry/c:mesh", ns):
        sources = {}
        for source in mesh.findall("c:source", ns):
            array = source.find("c:float_array", ns)
            accessor = source.find("c:technique_common/c:accessor", ns)
            sources[source.get("id")] = np.fromstring(array.text, sep=" ").reshape(-1, int(accessor.get("stride")))
        vertices = {v.get("id"): v.find("c:input[@semantic='POSITION']", ns).get("source")[1:] for v in mesh.findall("c:vertices", ns)}
        for faces in mesh.findall("c:triangles", ns):
            inputs = faces.findall("c:input", ns)
            width = 1 + max(int(x.get("offset", "0")) for x in inputs)
            vertex = next(x for x in inputs if x.get("semantic") == "VERTEX")
            indices = np.fromstring(faces.find("c:p", ns).text, sep=" ", dtype=np.int64).reshape(-1, 3, width)
            positions = sources[vertices[vertex.get("source")[1:]]]
            triangles.append(positions[indices[:, :, int(vertex.get("offset", "0"))], :3])
    if not triangles:
        raise ValueError(f"No triangular mesh geometry: {path}")
    return np.concatenate(triangles)


def project_faces(triangles, w2c, intrinsic, height, width):
    points = triangles.reshape(-1, 3)
    camera = (np.c_[points, np.ones(len(points))] @ w2c.T)[:, :3].reshape(-1, 3, 3)
    camera = camera[(camera[:, :, 2] > 1e-5).all(axis=1)]
    uv = camera[:, :, :2] / camera[:, :, 2:3]
    uv[:, :, 0] = uv[:, :, 0] * intrinsic[0, 0] + intrinsic[0, 2]
    uv[:, :, 1] = uv[:, :, 1] * intrinsic[1, 1] + intrinsic[1, 2]
    outside = ((uv[:, :, 0] < -width) | (uv[:, :, 0] > 2 * width) | (uv[:, :, 1] < -height) | (uv[:, :, 1] > 2 * height)).all(axis=1)
    return np.rint(uv[~outside]).astype(np.int32)


class RoboTwinMeshRenderer:
    def __init__(self, config_path, resize_to=(256, 320), background=190, gripper_gray=55):
        self.config_path = Path(config_path)
        self.config = _load_yaml(self.config_path)
        self.urdf = (self.config_path.parent / self.config["urdf_path"]).resolve()
        self.fk = _UrdfSerialFK(self.urdf, self.config["robot_pose"][0])
        self.height, self.width = resize_to
        self.background, self.gripper_gray = int(background), int(gripper_gray)
        self.meshes = {}
        self.assets = [self.config_path.resolve(), self.urdf]
        root = ET.parse(self.urdf).getroot()
        self.child_joint = {j.find("child").get("link"): j.get("name") for j in root.findall("joint")}
        for prefix in ("fl", "fr"):
            for number in (6, 7, 8):
                link = f"{prefix}_link{number}"
                visual = root.find(f"link[@name='{link}']/visual")
                mesh = visual.find("geometry/mesh")
                path = (self.urdf.parent / mesh.get("filename")).resolve()
                triangles = load_dae_triangles(path)
                scale = _parse_xyz(mesh.get("scale"), default=(1, 1, 1))
                origin = visual.find("origin")
                transform = _make_transform(_parse_xyz(origin.get("xyz")), _parse_xyz(origin.get("rpy")))
                points = triangles.reshape(-1, 3) * scale
                self.meshes[link] = (np.c_[points, np.ones(len(points))] @ transform.T)[:, :3].reshape(triangles.shape)
                self.assets.append(path)

    @lru_cache(maxsize=2)
    def episode(self, path):
        with h5py.File(path, "r") as f:
            # The source action array has the same length as the recorded images.
            from .robotwin import _action_source_length
            n = _action_source_length(f)
            left, lg, right, rg = _read_action_components(f, np.arange(n))
            actions = np.concatenate((left, lg[:, None], right, rg[:, None]), axis=1)
            cameras = {}
            for name in ("head_camera", "left_camera", "right_camera"):
                group = f["observation"][name]
                image = _decode_rgb_bytes(group["rgb"][0])
                cameras[name] = dict(intrinsic=np.asarray(group["intrinsic_cv"]), extrinsic=np.asarray(group["extrinsic_cv"]), size=image.shape[:2])
        return actions, cameras

    @lru_cache(maxsize=512)
    def frame(self, path, frame_id, camera_names):
        actions, cameras = self.episode(path)
        action = actions[frame_id]
        joints = dict(zip(self.config["arm_joints_name"][0], action[:6]))
        joints.update(zip(self.config["arm_joints_name"][1], action[7:13]))
        low, high = self.config["gripper_scale"]
        for index, spec in zip((6, 13), self.config["gripper_name"]):
            value = low + np.clip(action[index], 0, 1) * (high - low)
            joints[spec["base"]] = value
            for name, multiplier, offset in spec["mimic"]:
                joints[name] = value * multiplier + offset
        transforms = self.fk.forward(joints)
        triangles = []
        for name, local in self.meshes.items():
            transform = transforms[self.child_joint[name]]
            points = local.reshape(-1, 3)
            triangles.append((np.c_[points, np.ones(len(points))] @ transform.T)[:, :3].reshape(local.shape))
        images = []
        for name in camera_names:
            camera = cameras[name]
            intrinsic = camera["intrinsic"][min(frame_id, len(camera["intrinsic"]) - 1)]
            extrinsic = camera["extrinsic"][min(frame_id, len(camera["extrinsic"]) - 1)]
            intrinsic = _scale_intrinsic(intrinsic, *camera["size"], self.height, self.width)
            image = np.full((self.height, self.width, 3), self.background, np.uint8)
            # Match the reference's per-link partial fill plus all-face contours.
            # Combining both arms into one fill operation changes polygon parity.
            for mesh in triangles:
                faces = project_faces(mesh, _to_4x4_w2c(extrinsic), intrinsic, self.height, self.width)
                if len(faces):
                    mask = np.zeros((self.height, self.width), np.uint8)
                    cv2.fillPoly(mask, faces, 255, lineType=cv2.LINE_AA)
                    image[mask > 0] = self.gripper_gray
                    color = (self.gripper_gray,) * 3
                    cv2.polylines(image, faces, isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)
            images.append(image)
        return np.concatenate(images, axis=0)

    def render(self, meta):
        cameras = tuple(meta["camera_names"])
        return [Image.fromarray(self.frame(meta["path"], int(i), cameras)) for i in meta["frame_ids"]]
