"""Shared BEV layout construction utilities and CLI."""

import argparse
import json
import os
import pickle
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "work_dirs/matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib.style as mpl_style
from PIL import Image, ImageDraw
from pyquaternion import Quaternion
from tools.trajectory_generation.box_conventions import (
    convert_boxes, transform_boxes_between_lidars,
)

_ORIGINAL_STYLE_USE = mpl_style.use


def style_use_with_legacy_alias(style):
    if style == "seaborn-whitegrid":
        try:
            return _ORIGINAL_STYLE_USE(style)
        except OSError:
            return _ORIGINAL_STYLE_USE("seaborn-v0_8-whitegrid")
    return _ORIGINAL_STYLE_USE(style)


mpl_style.use = style_use_with_legacy_alias

from nuscenes.map_expansion.map_api import NuScenesMap


STATIC_CLASSES = [
    "drivable_area",
    "ped_crossing",
    "walkway",
    "stop_line",
    "carpark_area",
    "road_divider",
    "lane_divider",
    "road_block",
]

OBJECT_CLASSES = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]


DEFAULT_NUSCENES_CLASSES = (
    "car",
    "truck",
    "trailer",
    "bus",
    "construction_vehicle",
    "bicycle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "barrier",
)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def class_names_from_metainfo(metainfo):
    if "classes" in metainfo:
        return tuple(metainfo["classes"])
    if "categories" in metainfo:
        categories = metainfo["categories"]
        return tuple(
            name for name, _ in sorted(
                categories.items(), key=lambda item: int(item[1])))
    return DEFAULT_NUSCENES_CLASSES


def load_infos(path):
    data = load_pickle(path)
    if "data_list" in data:
        return data["data_list"], class_names_from_metainfo(
            data.get("metainfo", {}))
    if "infos" in data:
        return data["infos"], DEFAULT_NUSCENES_CLASSES
    raise KeyError(f"{path} must contain either data_list or infos.")


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def matrix_from_quat_translation(rotation, translation):
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = Quaternion(rotation).rotation_matrix
    mat[:3, 3] = np.asarray(translation, dtype=np.float64)
    return mat


def load_scene_locations(data_root, version):
    version_root = os.path.join(data_root, version)
    scene_path = os.path.join(version_root, "scene.json")
    log_path = os.path.join(version_root, "log.json")
    if not os.path.exists(scene_path) or not os.path.exists(log_path):
        raise FileNotFoundError(
            "Cannot load scene locations. Expected scene/log metadata at "
            f"{scene_path} and {log_path}.")

    logs = {
        log["token"]: log["location"]
        for log in load_json(log_path)
    }
    return {
        scene["name"]: logs[scene["log_token"]]
        for scene in load_json(scene_path)
    }


def get_info_location(info, scene_locations):
    if info.get("location") is not None:
        return info["location"]
    scene_name = info.get("scene_name")
    if scene_name in scene_locations:
        return scene_locations[scene_name]
    raise KeyError(
        "Cannot infer nuScenes map location for info "
        f"token={info.get('token')} scene_name={scene_name}.")


def get_lidar2ego(info):
    lidar_points = info.get("lidar_points", {})
    if "lidar2ego" in lidar_points:
        return np.asarray(lidar_points["lidar2ego"], dtype=np.float64)
    return matrix_from_quat_translation(
        info["lidar2ego_rotation"], info["lidar2ego_translation"])


def get_ego2global(info):
    if "ego2global" in info:
        return np.asarray(info["ego2global"], dtype=np.float64)
    return matrix_from_quat_translation(
        info["ego2global_rotation"], info["ego2global_translation"])


class BEVLayoutRenderer:
    def __init__(self, data_root, xbound=(-40, 40, 0.4),
                 ybound=(-40, 40, 0.4), box_convention='legacy'):
        if box_convention not in ('native', 'legacy'):
            raise ValueError(f'Unknown box convention: {box_convention}')
        self.box_convention = box_convention
        self.data_root = data_root
        self.patch_h = ybound[1] - ybound[0]
        self.patch_w = xbound[1] - xbound[0]
        self.canvas_h = int(self.patch_h / ybound[2])
        self.canvas_w = int(self.patch_w / xbound[2])
        self.canvas_size = (self.canvas_h, self.canvas_w)
        self.lidar2canvas = np.array(
            [[self.canvas_h / self.patch_h, 0, self.canvas_h / 2],
             [0, self.canvas_w / self.patch_w, self.canvas_w / 2],
             [0, 0, 1]],
            dtype=np.float64)
        self.map_cache = {}

    def render(self, info):
        static_label = self._project_static(info)
        dynamic_label = self._project_dynamic(info)
        return np.concatenate([static_label, dynamic_label], axis=0)

    def _get_map(self, location):
        if location not in self.map_cache:
            self.map_cache[location] = NuScenesMap(self.data_root, location)
        return self.map_cache[location]

    def _project_static(self, info):
        lidar2global = info["ego2global"] @ info["lidar2ego"]
        map_pose = lidar2global[:2, 3]
        patch_box = (map_pose[0], map_pose[1], self.patch_h, self.patch_w)
        forward = lidar2global[:3, :3] @ np.array([1, 0, 0])
        patch_angle = np.arctan2(forward[1], forward[0]) / np.pi * 180

        masks = self._get_map(info["location"]).get_map_mask(
            patch_box=patch_box,
            patch_angle=patch_angle,
            layer_names=STATIC_CLASSES,
            canvas_size=self.canvas_size,
        )
        return masks.transpose(0, 2, 1).astype(bool).astype(np.uint8)

    def _box_corners_xy(self, boxes):
        centers = boxes[:, :2]
        dx = boxes[:, 3]
        dy = boxes[:, 4]
        yaw = boxes[:, 6]
        local = np.stack([
            np.stack([dx / 2, dy / 2], axis=1),
            np.stack([dx / 2, -dy / 2], axis=1),
            np.stack([-dx / 2, -dy / 2], axis=1),
            np.stack([-dx / 2, dy / 2], axis=1),
        ], axis=1)
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        rot = np.stack([
            np.stack([cos_yaw, -sin_yaw], axis=1),
            np.stack([sin_yaw, cos_yaw], axis=1),
        ], axis=1)
        return local @ np.swapaxes(rot, 1, 2) + centers[:, None, :]

    def _project_dynamic(self, info):
        dynamic_mask = np.zeros(
            (len(OBJECT_CLASSES), self.canvas_h, self.canvas_w),
            dtype=np.uint8)
        boxes = convert_boxes(info['gt_boxes'],
                              info.get('box_convention', self.box_convention),
                              self.box_convention)
        names = info["gt_names"]
        if len(boxes) == 0:
            return dynamic_mask.transpose(0, 2, 1)

        for class_idx, class_name in enumerate(OBJECT_CLASSES):
            class_boxes = boxes[names == class_name]
            if len(class_boxes) == 0:
                continue
            corners = self._box_corners_xy(class_boxes)
            if self.box_convention == 'native':
                # Native [length,width,height,yaw] is already physical. Match
                # the explicit pixel-coordinate contract of the geometry audit.
                corners = (corners + np.array([self.patch_h, self.patch_w]) / 2) / np.array([
                    self.patch_h / self.canvas_h, self.patch_w / self.canvas_w])
            else:
                # Keep the historical 12 Hz convention and old checkpoint
                # condition construction unchanged unless native is requested.
                corners = np.pad(
                    corners,
                    ((0, 0), (0, 0), (0, 1)),
                    constant_values=1.0) @ self.lidar2canvas.T
                corners = corners[..., :2]
                centers = np.mean(corners, axis=1, keepdims=True)
                corners[:, :, 1] = 2 * centers[:, :, 1] - corners[:, :, 1]

            render = Image.fromarray(dynamic_mask[class_idx])
            draw = ImageDraw.Draw(render)
            for polygon in corners:
                draw.polygon(
                    polygon.round().astype(np.int32).flatten().tolist(),
                    fill=1)
            dynamic_mask[class_idx] = np.array(render)

        return dynamic_mask.transpose(0, 2, 1)


def render_layout_sequence(renderer, future_poses, target_lidar2ego,
                           future_render_infos, location):
    """Render future frames using their ego poses and the target LiDAR rig."""
    layouts = []
    for future_pose, source_info in zip(future_poses, future_render_infos):
        future_lidar2global = future_pose @ target_lidar2ego
        source_lidar2global = source_info['ego2global'] @ source_info['lidar2ego']
        boxes = transform_boxes_between_lidars(
            convert_boxes(source_info['gt_boxes'],
                          source_info.get('box_convention', 'native'), 'native'),
            source_lidar2global, future_lidar2global)
        render_info = dict(
            token=source_info['token'], location=location,
            lidar2ego=target_lidar2ego, ego2global=future_pose,
            gt_boxes=boxes, gt_names=source_info['gt_names'],
            box_convention='native',
        )
        layouts.append(renderer.render(render_info).astype(np.uint8))
    return np.stack(layouts, axis=0)


def current_to_future_transforms(current_pose, future_poses):
    """Map current-ego coordinates into each future ego frame."""
    return np.stack([
        np.linalg.inv(future_pose) @ current_pose
        for future_pose in future_poses
    ], axis=0)


def instances_to_boxes_and_names(info, class_names):
    if "instances" not in info:
        raise KeyError(
            f"Keyframe info token={info.get('token')} must contain "
            "instances for v2-format annotations.")

    boxes = []
    names = []
    for instance in info["instances"]:
        if not bool(instance.get("bbox_3d_isvalid", True)):
            continue
        if "bbox_3d" not in instance:
            raise KeyError(
                f"Instance in token={info.get('token')} is missing bbox_3d.")
        label = int(instance.get(
            "bbox_label_3d", instance.get("bbox_label", -1)))
        if label < 0 or label >= len(class_names):
            continue
        bbox = np.asarray(instance["bbox_3d"], dtype=np.float64)
        if bbox.shape != (7,):
            raise ValueError(
                f"Instance bbox_3d in token={info.get('token')} must have "
                f"shape (7,), got {bbox.shape}.")
        boxes.append(bbox)
        names.append(class_names[label])

    gt_boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 7)
    gt_names = np.asarray(names)
    return gt_boxes, gt_names


def keyframe_info_to_render_info(info, scene_locations, class_names,
                                 box_convention='native'):
    if "gt_boxes" in info and "gt_names" in info:
        mask = info.get(
            "valid_flag",
            np.ones(len(info["gt_boxes"]), dtype=bool))
        gt_boxes = np.asarray(info["gt_boxes"], dtype=np.float64)[mask].copy()
        gt_names = np.asarray(info["gt_names"])[mask].copy()
    else:
        gt_boxes, gt_names = instances_to_boxes_and_names(info, class_names)

    gt_boxes = convert_boxes(gt_boxes, box_convention, 'native')

    return dict(
        token=info["token"],
        location=get_info_location(info, scene_locations),
        lidar2ego=get_lidar2ego(info),
        ego2global=get_ego2global(info),
        gt_boxes=gt_boxes,
        gt_names=gt_names,
        box_convention='native',
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate keyframe nuScenes BEV layout npy files.")
    parser.add_argument(
        "--ann-file",
        default="data/nus_infos/nuscenes_infos_train.pkl",
        help="Keyframe ann pkl with a data_list field.")
    parser.add_argument("--data-root", default="data/nuscenes")
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument(
        "--save-root",
        default="work_dirs/bev_layout_keyframes_physical_v1")
    parser.add_argument('--box-convention', choices=('native', 'legacy'), default='native',
                        help='Native keyframe annotations use length,width,yaw; legacy retains the old local reflection')
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def save_or_render(renderer, render_info, args):
    save_path = os.path.join(args.save_root, render_info["token"] + ".npy")
    if os.path.exists(save_path) and not args.overwrite:
        return "skipped"

    bev_layout = renderer.render(render_info).astype(np.uint8)
    np.save(save_path, bev_layout)
    return "rendered"


def main():
    args = parse_args()
    infos, class_names = load_infos(args.ann_file)
    if args.limit is not None:
        infos = infos[:args.limit]

    scene_locations = load_scene_locations(args.data_root, args.version)
    renderer = BEVLayoutRenderer(args.data_root, box_convention=args.box_convention)
    os.makedirs(args.save_root, exist_ok=True)
    convention_path = os.path.join(args.save_root, 'layout_convention.json')
    if os.path.exists(convention_path):
        stored = load_json(convention_path).get('box_convention')
        if stored != args.box_convention:
            raise ValueError('Output directory has a different box convention; use a new directory')
    elif (args.box_convention == 'native' and not args.overwrite
          and any(name.endswith('.npy') for name in os.listdir(args.save_root))):
        raise ValueError('Cannot reuse unverified existing layouts; choose a fresh directory')

    counts = dict(generated=0, rendered=0, skipped=0)
    for idx, info in enumerate(infos, 1):
        render_info = keyframe_info_to_render_info(
            info, scene_locations, class_names)
        status = save_or_render(renderer, render_info, args)
        counts[status] += 1
        if status != "skipped":
            counts["generated"] += 1
        if idx % 50 == 0 or idx == len(infos):
            print(
                f"processed {idx}/{len(infos)} "
                f"generated={counts['generated']} "
                f"rendered={counts['rendered']} "
                f"skipped={counts['skipped']}")
    with open(convention_path, 'w') as stream:
        json.dump(dict(box_convention=args.box_convention, annotation=args.ann_file), stream, indent=2)


if __name__ == "__main__":
    main()
