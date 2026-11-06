#!/usr/bin/env python3
"""Render a saved generate_scene.py frame or full GIF without loading the model.

Example (headless software rendering):
    EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=true conda run -n loworld \
        python tools/visualization/scene.py \
        --predictions work_dirs/visualization/scene_0001/predictions.npz \
        --output work_dirs/visualization/scene_0001/preview/frame_000.png

Set --output to a .gif path to render the entire sequence at --fps (default 12).
Use --keep-frames to retain PNGs beside the GIF in <output_stem>/frames.
Otherwise temporary raw frame storage is removed after GIF encoding.

OCC indices follow the tokenizer's (X, Y, Z) convention. Saved LiDAR points
are transformed to ego coordinates using the corresponding saved extrinsic.
Only exposed voxel faces are drawn; labels and point positions are unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from functools import lru_cache
from contextlib import ExitStack
import tempfile
import time
from pathlib import Path

# Configure headless EGL before Open3D initializes its rendering backend.
# Use hardware rendering when a device is exposed; otherwise use Mesa software.
if not os.environ.get('DISPLAY'):
    os.environ.setdefault('EGL_PLATFORM', 'surfaceless')
    if not Path('/dev/nvidiactl').exists() and not list(Path('/dev/dri').glob('renderD*')):
        os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE', 'true')

import cv2
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.visualization.colors import get_occ3d_color_map
from tools.genie.occ_generation.scene_runtime import atomic_file, sha256_file, write_json
from tools.visualization.gif import encode_gif


ORIGIN = np.array([-40.0, -40.0, -1.0])
VOXEL_SIZE = 0.4
BACKGROUND = (255, 255, 255)
DEFAULT_EGO_LIFT_M = 1.5
CAMERA_EYE = [-65, -38, 74]
CAMERA_TARGET = [2, 0, 0]
CAMERA_UP = [0, 0, 1]
CAMERA_HALF_WIDTH = 53.0
# The shared helper returns BGR; Open3D and PIL require RGB.
PALETTE = get_occ3d_color_map()[:, ::-1].astype(np.float64) / 255.0


def srgb_to_linear(colors):
    """Open3D vertex colors enter the lighting pipeline in linear RGB."""
    return np.where(colors <= 0.04045, colors / 12.92,
                    ((colors + 0.055) / 1.055) ** 2.4)


def surface_mesh(labels):
    """Build the exact union boundary of all occupied 0.4 m cubes."""
    occupied = (labels >= 0) & (labels < 17)
    padded = np.pad(occupied, 1)
    vertices, triangles, colors = [], [], []
    # Outward counterclockwise quads, with coordinates relative to voxel corner.
    faces = [
        (0, -1, [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]]),
        (0, 1, [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]]),
        (1, -1, [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]]),
        (1, 1, [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]]),
        (2, -1, [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]]),
        (2, 1, [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]),
    ]
    count = 0
    for axis, direction, corners in faces:
        slices = [slice(1, n + 1) for n in labels.shape]
        slices[axis] = slice(1 + direction, labels.shape[axis] + 1 + direction)
        exposed = occupied & ~padded[tuple(slices)]
        indices = np.argwhere(exposed)
        if not len(indices):
            continue
        xyz = ORIGIN + (indices[:, None, :] + np.array(corners)) * VOXEL_SIZE
        vertices.append(xyz.reshape(-1, 3))
        offsets = np.arange(len(indices))[:, None, None] * 4 + count
        triangles.append((offsets + np.array([[0, 1, 2], [0, 2, 3]]))
                         .reshape(-1, 3))
        colors.append(np.repeat(PALETTE[labels[tuple(indices.T)]], 4, axis=0))
        count += 4 * len(indices)
    mesh = o3d.geometry.TriangleMesh()
    if vertices:
        mesh.vertices = o3d.utility.Vector3dVector(np.concatenate(vertices))
        mesh.triangles = o3d.utility.Vector3iVector(np.concatenate(triangles))
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            srgb_to_linear(np.concatenate(colors)))
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
    return mesh


def height_colors(height):
    # Match projects/visualizer/lidar_visualizer.py: OpenCV Viridis by height.
    if len(height) == 0:
        return np.empty((0, 3), dtype=np.float64)
    levels = (np.clip((height + 1.0) / 6.4, 0, 1) * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(levels.reshape(-1, 1), cv2.COLORMAP_VIRIDIS)
    return bgr[:, 0, ::-1].astype(np.float64) / 255.0


def material(shader, color=None):
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = shader
    mat.base_roughness = 0.95
    if color is not None:
        mat.base_color = [*color, 1.0]
    return mat


def add_reference(scene, ego_lift_m=DEFAULT_EGO_LIFT_M):
    vertices, lines = [], []
    for t in range(-40, 41, 10):
        n = len(vertices)
        vertices.extend([[t, -40, -1.04], [t, 40, -1.04],
                         [-40, t, -1.04], [40, t, -1.04]])
        lines.extend([[n, n + 1], [n + 2, n + 3]])
    grid = o3d.geometry.LineSet(
        o3d.utility.Vector3dVector(vertices), o3d.utility.Vector2iVector(lines))
    grid.paint_uniform_color(srgb_to_linear(np.array([0.89, 0.90, 0.92])))
    mat = material('unlitLine')
    mat.line_width = 1.0
    scene.add_geometry('10m_reference_grid', grid, mat)

    # Lift the display marker above voxelized road surfaces. This translation
    # applies only to the reference vehicle and its arrow, never the predictions.
    marker_offset = np.array([0.0, 0.0, ego_lift_m])
    for name, dimensions, position, color in [
        ('ego_body', [4.084, 1.85, 0.85], [-2.042, -0.925, -0.75],
         [0.92, 0.94, 0.98]),
        ('ego_cabin', [2.1, 1.55, 0.55], [-1.2, -0.775, 0.1],
         [0.30, 0.49, 0.66]),
    ]:
        mesh = o3d.geometry.TriangleMesh.create_box(*dimensions)
        mesh.translate(np.asarray(position) + marker_offset)
        mesh.compute_vertex_normals()
        scene.add_geometry(name, mesh, material('defaultLit', color))
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.10, cone_radius=0.35,
        cylinder_height=3.2, cone_height=0.9)
    arrow.rotate(arrow.get_rotation_matrix_from_xyz([0, np.pi / 2, 0]),
                 center=[0, 0, 0])
    arrow.translate(np.array([3.0, 0, -0.55]) + marker_offset)
    arrow.compute_vertex_normals()
    scene.add_geometry('ego_forward', arrow,
                       material('defaultUnlit', [1.0, 0.71, 0.29]))


def build_renderer(width, height, ego_lift_m=DEFAULT_EGO_LIFT_M):
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    scene = renderer.scene
    scene.set_background(np.array([*(v / 255 for v in BACKGROUND), 1.0]))
    scene.set_lighting(scene.LightingProfile.SOFT_SHADOWS, [-0.4, -0.4, -1])
    scene.scene.set_indirect_light_intensity(30000)
    scene.view.set_ambient_occlusion(True)
    scene.view.set_antialiasing(True)
    scene.view.set_shadowing(False)
    scene.view.set_color_grading(o3d.visualization.rendering.ColorGrading(
        o3d.visualization.rendering.ColorGrading.Quality.ULTRA,
        o3d.visualization.rendering.ColorGrading.ToneMapping.LINEAR))
    add_reference(scene, ego_lift_m)
    renderer.setup_camera(50.0, CAMERA_TARGET, CAMERA_EYE, CAMERA_UP)
    half_width = CAMERA_HALF_WIDTH
    half_height = half_width * height / width
    scene.camera.set_projection(
        o3d.visualization.rendering.Camera.Projection.Ortho,
        -half_width, half_width, -half_height, half_height, 1.0, 250.0)
    return renderer


def render_panels(labels, points, width, height, renderer=None):
    if renderer is None:
        renderer = build_renderer(width, height)
    scene = renderer.scene
    mesh = surface_mesh(labels)
    if len(mesh.vertices):
        scene.add_geometry('occ_surface', mesh, material('defaultLit'))
    occ_image = np.asarray(renderer.render_to_image()).copy()
    scene.remove_geometry('occ_surface')
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud.colors = o3d.utility.Vector3dVector(
        srgb_to_linear(height_colors(points[:, 2])))
    mat = material('defaultUnlit')
    mat.point_size = 3.3 * width / 1160
    scene.add_geometry('lidar', cloud, mat)
    lidar_image = np.asarray(renderer.render_to_image()).copy()
    scene.remove_geometry('lidar')
    return occ_image, lidar_image, dict(
        voxel_count=int(((labels >= 0) & (labels < 17)).sum()),
        surface_triangles=len(mesh.triangles), displayed_points=len(points),
        camera_eye=[-65, -38, 74], camera_target=[2, 0, 0],
        projection='orthographic', horizontal_span_m=106.0)


@lru_cache(maxsize=8)
def font(size):
    return ImageFont.truetype(
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size)


def compose(occ_image, lidar_image):
    """Place only the legends inside the upper left of each rendered panel."""
    width, panel_height = occ_image.shape[1], occ_image.shape[0]
    gap = 24
    right = width + gap
    canvas = Image.new('RGB', (2 * width + gap, panel_height), BACKGROUND)
    canvas.paste(Image.fromarray(occ_image), (0, 0))
    canvas.paste(Image.fromarray(lidar_image), (right, 0))
    draw = ImageDraw.Draw(canvas)
    muted = '#59677a'
    items = [('Road', 11), ('Sidewalk', 13), ('Terrain', 14),
             ('Structure', 15), ('Vegetation', 16), ('Vehicle', 4), ('Person', 7)]
    for index, (title, cls) in enumerate(items):
        x = 20 + (index % 4) * 148
        y = 20 + (index // 4) * 30
        color = tuple((PALETTE[cls] * 255).astype(int))
        draw.rounded_rectangle((x, y + 4, x + 17, y + 21), radius=4, fill=color)
        draw.text((x + 27, y), title, font=font(18), fill=muted)
    # The color bar shares the fixed height limits used for every point/frame.
    bar_width = 360
    bar = (height_colors(np.linspace(-1, 5.4, bar_width)) * 255).astype(np.uint8)
    x, y = right + 20, 24
    canvas.paste(Image.fromarray(np.repeat(bar[None], 12, axis=0)), (x, y))
    for label, fraction in [('-1.0 m', 0), ('+2.2 m', 0.5), ('+5.4 m', 1)]:
        label_width = draw.textlength(label, font=font(18))
        draw.text((x + fraction * (bar_width - label_width), y + 21),
                  label, font=font(18), fill=muted)
    return canvas


def load_frame(data, frame):
    if not 0 <= frame <= len(data['future_tokens_12hz']):
        raise ValueError('Frame is outside the saved sequence')
    if str(data['point_coordinate']) != 'lidar':
        raise ValueError('Expected saved points in lidar coordinates')
    if frame == 0:
        labels = data['current_occ']
        points = data['current_points_xyz']
        transform = data['point_lidar2ego']
    else:
        i = frame - 1
        labels = data['pred_occ_12hz'][i]
        start, stop = data['pred_points_xyz_offsets'][i:i + 2]
        points = data['pred_points_xyz_concat'][start:stop]
        transform = data['pred_points_lidar2ego'][i]
    if labels.shape != (200, 200, 16):
        raise ValueError(f'Expected 200x200x16 OCC grid, got {labels.shape}')
    points = points @ transform[:3, :3].T + transform[:3, 3]
    keep = (np.isfinite(points).all(axis=1) &
            (points >= ORIGIN).all(axis=1) &
            (points < ORIGIN + np.array(labels.shape) * VOXEL_SIZE).all(axis=1))
    return labels, points[keep]


def render_saved_predictions(predictions, output, *, frame=0, fps=12,
                             ego_lift=DEFAULT_EGO_LIFT_M, keep_frames=False,
                             metadata=None, frames_dir=None, run_id=None):
    """Render with one renderer, one palette, and optional persistent PNGs."""
    started = time.perf_counter()
    timings = {}
    predictions, output = Path(predictions), Path(output)
    source_sha256 = sha256_file(predictions)
    if not 1 <= fps <= 100:
        raise ValueError('--fps must be between 1 and 100')
    if not np.isfinite(ego_lift) or ego_lift < 0:
        raise ValueError('--ego-lift must be finite and nonnegative')
    output.parent.mkdir(parents=True, exist_ok=True)
    animation = output.suffix.lower() == '.gif'
    if animation and frame != 0:
        raise ValueError('--frame is only supported for a single PNG')
    start = time.perf_counter()
    keys = ['future_tokens_12hz', 'point_coordinate']
    if animation or frame == 0:
        keys += ['current_occ', 'current_points_xyz', 'point_lidar2ego']
    if animation or frame > 0:
        keys += ['pred_occ_12hz', 'pred_points_xyz_concat',
                 'pred_points_xyz_offsets', 'pred_points_lidar2ego']
    with np.load(predictions, allow_pickle=False) as archive:
        data = {key: archive[key] for key in keys}
    timings['load_predictions'] = time.perf_counter() - start
    indices = list(range(len(data['future_tokens_12hz']) + 1)) if animation else [frame]
    frame_paths, frame_stats = [], []
    start = time.perf_counter()
    with ExitStack() as stack:
        storage = None
        if animation:
            if keep_frames:
                frames_dir = Path(frames_dir) if frames_dir else output.with_suffix('') / 'frames'
                frames_dir.mkdir(parents=True, exist_ok=True)
            else:
                scratch = Path(stack.enter_context(tempfile.TemporaryDirectory(
                    prefix='.scene_frames_', dir=output.parent)))
                # Raw memory mapping avoids compressing and rereading 81 PNGs.
                storage = np.memmap(scratch / 'rgb.raw', mode='w+', dtype=np.uint8,
                                    shape=(len(indices), 1050, 2344, 3))
                stack.callback(storage._mmap.close)
        renderer = build_renderer(1160, 1050, ego_lift_m=ego_lift)
        for index in indices:
            labels, points = load_frame(data, index)
            occ_image, lidar_image, stats = render_panels(
                labels, points, 1160, 1050, renderer=renderer)
            image = compose(occ_image, lidar_image)
            if animation and not keep_frames:
                storage[index] = np.asarray(image)
                frame_paths.append(storage[index])
            else:
                destination = frames_dir / f'frame_{index:06d}.png' if animation else output
                image.save(destination)
                frame_paths.append(destination)
            frame_stats.append(dict(frame=index, **stats))
            print(f'Rendered frame {index:02d}/{indices[-1]:02d}', flush=True)
        timings['render_frames'] = time.perf_counter() - start
        stats = dict(frames=frame_stats) if animation else frame_stats[0]
        if animation:
            print('Encoding GIF with shared palette...', flush=True)
            start = time.perf_counter()
            stats['animation'] = encode_gif(frame_paths, output, fps)
            timings['encode_and_verify_gif'] = time.perf_counter() - start
        frame_paths.clear()
    stats.update(source=str(predictions), source_sha256=source_sha256, run_id=run_id,
                 voxel_size_m=VOXEL_SIZE, height_range_m=[-1.0, 5.4],
                 camera=dict(projection='orthographic', eye=CAMERA_EYE,
                             target=CAMERA_TARGET, up=CAMERA_UP,
                             horizontal_bounds=[-CAMERA_HALF_WIDTH, CAMERA_HALF_WIDTH],
                             near=1.0, far=250.0),
                 ego_marker_lift_m=ego_lift, background='white',
                 layout='no titles or footer; legends inside upper left of each panel',
                 occ_palette='tools.visualization.colors.get_occ3d_color_map',
                 lidar_palette='cv2.COLORMAP_VIRIDIS (repository LiDAR visualizer)',
                 source_points_unchanged=True, display_filter='finite, inside OCC range',
                 keep_frames=keep_frames, stages_seconds=timings,
                 wall_seconds=time.perf_counter() - started)
    if sha256_file(predictions) != source_sha256:
        raise RuntimeError('Prediction source changed while rendering')
    write_json(Path(metadata) if metadata else output.with_suffix('.json'), stats)
    print(f'Saved {output} ({stats["wall_seconds"]:.2f} s)', flush=True)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frame', type=int, default=0,
                        help='0=current; 1..80=future at 12 Hz; PNG mode only')
    parser.add_argument('--fps', type=int, default=12, help='GIF playback FPS')
    parser.add_argument('--ego-lift', type=float, default=DEFAULT_EGO_LIFT_M,
                        help='Vertical offset in meters for the ego display marker')
    parser.add_argument('--keep-frames', action='store_true',
                        help='Keep full-resolution PNG frames; default uses temporary raw storage')
    parser.add_argument('--metadata', type=Path, help='JSON metadata destination')
    parser.add_argument('--frames-dir', type=Path, help='Persistent PNG directory with --keep-frames')
    parser.add_argument('--run-id', help='Parent scene run identifier')
    args = parser.parse_args()
    render_saved_predictions(args.predictions, args.output, frame=args.frame,
                             fps=args.fps, ego_lift=args.ego_lift, keep_frames=args.keep_frames,
                             metadata=args.metadata, frames_dir=args.frames_dir, run_id=args.run_id)


if __name__ == '__main__':
    main()
