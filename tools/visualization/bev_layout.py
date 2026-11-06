"""Visualize saved scene layout conditions without rerunning inference.

Run with the loworld conda environment. Annotation outlines are an optional
geometric reference; no layout values or prediction files are modified.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.visualization.gif import encode_gif


CLASSES = (
    'drivable_area', 'ped_crossing', 'walkway', 'stop_line',
    'carpark_area', 'road_divider', 'lane_divider', 'road_block',
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
)
COLORS = np.array([
    (211, 220, 230), (255, 201, 151), (151, 211, 158), (237, 224, 144),
    (187, 174, 218), (240, 165, 40), (255, 255, 255), (177, 188, 202),
    (0, 150, 245), (160, 32, 240), (0, 190, 190), (245, 205, 0),
    (135, 60, 0), (255, 120, 50), (225, 105, 0), (245, 135, 193),
    (240, 35, 45), (225, 180, 65),
], dtype=np.uint8)
FONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
PANEL = 600
TOP = 95
BOTTOM = 176
MARGIN = 38
GAP = 24


def font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default(size=size)


def colorize(layout, start=0, stop=18):
    image = np.full((200, 200, 3), 248, dtype=np.uint8)
    # All channels are multi-label; render in channel order, matching the
    # existing renderer's layering. Individual channels are also exported.
    for channel in range(start, stop):
        image[layout[channel] > 0] = COLORS[channel]
    # Raw layout axes are [lidar X, lidar Y]. This display is forward up,
    # right right: rows = -lidar Y, columns = +lidar X.
    return Image.fromarray(np.rot90(image, 1)).resize(
        (PANEL, PANEL), Image.Resampling.NEAREST)


def boxes_for_frame(info, target_lidar2ego, legacy_boxes=False):
    if info is None:
        return []
    source_lidar2ego = np.asarray(info['lidar_points']['lidar2ego'])
    # Source and candidate use the same ego pose, as in render_future_layouts.
    source_to_candidate = np.linalg.inv(target_lidar2ego) @ source_lidar2ego
    boxes = []
    for instance in info.get('instances', []):
        if not instance.get('bbox_3d_isvalid', True):
            continue
        label = int(instance.get('bbox_label_3d', instance.get('bbox_label', -1)))
        if label < 0:
            continue
        box = np.asarray(instance['bbox_3d'], dtype=float)
        if legacy_boxes:
            box = box.copy()
            box[[3, 4]] = box[[4, 3]]
            box[6] = -box[6] - np.pi / 2
        dx, dy = box[3:5] / 2
        corners = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]])
        c, s = np.cos(box[6]), np.sin(box[6])
        corners = corners @ np.array([[c, s], [-s, c]]) + box[:2]
        points = np.column_stack([corners, np.full(4, box[2]), np.ones(4)])
        corners = (points @ source_to_candidate.T)[:, :2]
        boxes.append(corners)
    return boxes


def reference_convention(payloads, infos, target):
    """Infer the 12 Hz box convention using the same-token native annotation.

    The distributed 12 Hz infos retain the old [w,l,h,-yaw-pi/2] convention;
    native 2 Hz infos use [l,w,h,yaw]. Both are named bbox_3d, so checking only
    the field name would produce misleading reference outlines.
    """
    for payload in sorted(payloads, key=lambda item: len(item['infos'])):
        native = next((row for row in payload['infos'] if row['token'] == target), None)
        if native is None or len(payload['infos']) >= len(infos):
            continue
        for first in infos[target].get('instances', []):
            a = np.asarray(first['bbox_3d'], dtype=float)
            if np.isclose(a[3], a[4]):
                continue
            for second in native.get('instances', []):
                b = np.asarray(second['bbox_3d'], dtype=float)
                if first['bbox_label_3d'] != second['bbox_label_3d']:
                    continue
                if not np.allclose(a[:3], b[:3], atol=1e-4):
                    continue
                angle_error = lambda x: abs(np.arctan2(np.sin(x), np.cos(x)))
                if (np.allclose(a[3:6], b[[4, 3, 5]], atol=1e-4)
                        and angle_error(-a[6] - np.pi / 2 - b[6]) < 1e-4):
                    return True
                if (np.allclose(a[3:6], b[3:6], atol=1e-4)
                        and angle_error(a[6] - b[6]) < 1e-4):
                    return False
        raise ValueError('Cannot reconcile native and 12 Hz annotation box conventions')
    raise ValueError('Need a same-token native cache to validate reference box convention')


def panel_annotations(image, boxes=None):
    draw = ImageDraw.Draw(image)
    scale = PANEL / 80
    if boxes is not None:
        for corners in boxes:
            pixels = [(float((x + 40) * scale), float((40 - y) * scale))
                      for x, y in corners]
            draw.line(pixels + pixels[:1], fill=(25, 25, 25), width=2)
    # Cross and arrows mark the LiDAR origin, not the ego center. The mount is
    # about 0.94 m forward of the ego origin in scene-0007.
    mid = PANEL // 2
    draw.line((mid - 7, mid, mid + 7, mid), fill=(20, 20, 20), width=2)
    draw.line((mid, mid - 7, mid, mid + 7), fill=(20, 20, 20), width=2)
    for meter in (-40, -20, 0, 20, 40):
        pixel = min(PANEL - 1, max(0, round((meter + 40) * scale)))
        draw.line((pixel, PANEL - 6, pixel, PANEL - 1), fill=(35, 35, 35))
        draw.line((0, pixel, 5, pixel), fill=(35, 35, 35))
    draw.rectangle((10, 10, 201, 44), fill=(255, 255, 255))
    draw.text((16, 15), 'FRONT / LiDAR +Y', fill=(20, 20, 20), font=font(18))
    draw.line((PANEL - 110, PANEL - 25, PANEL - 35, PANEL - 25),
              fill=(25, 25, 25), width=3)
    draw.text((PANEL - 96, PANEL - 51), '10 m', fill=(25, 25, 25), font=font(17))
    return image


def render_frame(layout, frame_index, count, time_seconds, has_reference, boxes, scene_name):
    width = 2 * MARGIN + 3 * PANEL + 2 * GAP
    image = Image.new('RGB', (width, TOP + PANEL + BOTTOM), 'white')
    draw = ImageDraw.Draw(image)
    draw.text((MARGIN, 12), f'{scene_name}  |  Saved BEV layout conditions',
              fill=(20, 30, 40), font=font(25))
    kind = 'current context' if frame_index == 0 else 'future condition'
    draw.text((width - 630, 18),
              f'Frame {frame_index:03d}/{count - 1:03d}   t={time_seconds:.3f}s   {kind}',
              fill=(50, 60, 70), font=font(18))
    titles = ['Static map: channels 0-7', 'Map + objects: channels 0-17',
              'Objects + annotation outlines' if has_reference else 'Objects: channels 8-17']
    for index, (start, stop) in enumerate(((0, 8), (0, 18), (8, 18))):
        left = MARGIN + index * (PANEL + GAP)
        draw.text((left, 65), titles[index], fill=(30, 40, 50), font=font(20))
        panel = panel_annotations(colorize(layout, start, stop),
                                  boxes if index == 2 and has_reference else None)
        image.paste(panel, (left, TOP))
        draw.rectangle((left, TOP, left + PANEL, TOP + PANEL),
                       outline=(180, 185, 190), width=1)
    for channel, name in enumerate(CLASSES):
        col, row = channel % 6, channel // 6
        x, y = MARGIN + col * 314, TOP + PANEL + 18 + row * 31
        draw.rectangle((x, y + 3, x + 18, y + 21), fill=tuple(COLORS[channel]))
        draw.text((x + 26, y), name, fill=(35, 40, 45), font=font(17))
    draw.text((MARGIN, TOP + PANEL + 123),
              '80 x 80 m / 0.4 m cells. LiDAR origin = cross. FRONT up, RIGHT right. Black outlines = same-frame source boxes.',
              fill=(70, 75, 80), font=font(17))
    draw.text((MARGIN, TOP + PANEL + 148),
              'Multi-label map layers overlap; later channels cover earlier ones. Display rotation only; source arrays are unchanged.',
              fill=(70, 75, 80), font=font(16))
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path,
                        help='Explicit completed run; otherwise use latest successful run')
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--current-layout', type=Path,
                        help='Explicit corrected-context preview; does not modify saved predictions')
    parser.add_argument('--fps', type=float, default=12)
    args = parser.parse_args()
    if not 1 <= args.fps <= 100:
        parser.error('--fps must be between 1 and 100')
    scene = args.scene_dir.resolve()
    run = (args.run_dir or scene / 'latest').resolve()
    if not (run / 'manifest.json').is_file():
        raise FileNotFoundError('No successful scene run is available yet; wait for generation or use --run-dir')
    output = args.out_dir or scene / 'bevlayout'
    output.mkdir(parents=True, exist_ok=True)
    source = run / 'predictions/sequence.npz'
    scene_name = json.loads((run / 'manifest.json').read_text())['scene']
    with np.load(source, allow_pickle=False) as arrays:
        future = arrays['bev_layout']
        target = str(arrays['target_token'])
        tokens = [str(value) for value in arrays['future_tokens_12hz']]
        lidar2ego = arrays['point_lidar2ego']
        saved_current = arrays['current_bev_layout'] if 'current_bev_layout' in arrays else None
    if future.shape != (len(tokens), 18, 200, 200):
        raise ValueError(f'Unexpected layout shape: {future.shape}')
    current_path = args.current_layout or scene / '.cache/bev_layout' / f'{target}.npy'
    if args.current_layout is None and saved_current is not None:
        current = saved_current
        current_source = f'{source}[current_bev_layout]'
    elif current_path.exists():
        current = np.load(current_path, allow_pickle=False)
        current_source = str(current_path)
    else:
        raise FileNotFoundError(f'Current context layout missing: {current_path}')
    if current.shape == (200, 200, 18):
        current = np.moveaxis(current, -1, 0)
    layouts = np.concatenate((current[None], future), axis=0)
    tokens = [target] + tokens
    infos = {}
    annotation_sources = []
    payloads = []
    # Only read the scene cache created by the existing scene generator.
    for path in sorted((scene / '.cache').glob('annotations_*.pkl')):
        with path.open('rb') as stream:
            payload = pickle.load(stream)
        payloads.append(payload)
        rows = payload['infos']
        row_tokens = {row['token'] for row in rows}
        if not infos and all(token in row_tokens for token in tokens):
            infos = {row['token']: row for row in rows}
            annotation_sources = [str(path)]
    legacy_boxes = reference_convention(payloads, infos, target) if infos else False
    with (run / 'predictions/frames.csv').open() as stream:
        frame_rows = list(csv.DictReader(stream))
    if [row['token'] for row in frame_rows] != tokens:
        raise ValueError('Prediction frame tokens do not match layout tokens')
    times = [float(row['relative_time_seconds']) for row in frame_rows]
    frames = []
    for index, (layout, token, seconds) in enumerate(zip(layouts, tokens, times)):
        boxes = boxes_for_frame(infos.get(token), lidar2ego, legacy_boxes)
        frames.append(render_frame(layout, index, len(layouts), seconds,
                                   bool(infos), boxes, scene_name))
    selected = sorted(index for index in {0, 1, 6, 12, 24, 36, 48, 60, len(frames) - 1}
                      if index < len(frames))
    thumbs = []
    for index in selected:
        frames[index].save(output / f'frame_{index:06d}.png')
        # Combined panel at full size, with the timeline label above it.
        tile = Image.new('RGB', (PANEL, PANEL + 45), 'white')
        tile.paste(frames[index].crop((MARGIN + PANEL + GAP, TOP,
                                      MARGIN + 2 * PANEL + GAP, TOP + PANEL)), (0, 45))
        ImageDraw.Draw(tile).text((8, 9), f'Frame {index:03d} / t={times[index]:.3f}s',
                                  fill=(25, 35, 45), font=font(22))
        thumbs.append(tile)
    sheet = Image.new('RGB', (3 * PANEL, 3 * (PANEL + 45)), 'white')
    for index, tile in enumerate(thumbs):
        sheet.paste(tile, ((index % 3) * PANEL, (index // 3) * (PANEL + 45)))
    sheet.save(output / 'keyframes.png')
    animation = encode_gif(frames, output / 'bevlayout.gif', args.fps,
                           palette=frames[0].quantize(colors=256))
    channels = Image.new('RGB', (6 * 300, 3 * 340), 'white')
    for channel, name in enumerate(CLASSES):
        mask = np.full((200, 200, 3), 248, dtype=np.uint8)
        mask[layouts[1, channel] > 0] = COLORS[channel]
        tile = Image.fromarray(np.rot90(mask, 1)).resize((300, 300), Image.Resampling.NEAREST)
        x, y = (channel % 6) * 300, (channel // 6) * 340
        channels.paste(tile, (x, y + 40))
        ImageDraw.Draw(channels).text((x + 6, y + 10), f'{channel}: {name}',
                                      fill=(20, 30, 40), font=font(16))
    channels.save(output / 'channels_frame_000001.png')
    metadata = dict(
        source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        current_layout=current_source, corrected_context_preview=args.current_layout is not None,
        annotation_sources=annotation_sources,
        reference_box_conversion='swap width/length; yaw=-yaw-pi/2' if legacy_boxes else 'identity',
        frames=len(frames), fps=args.fps, duration_ms=animation['duration_ms'],
        future_shape=list(future.shape), classes=list(CLASSES),
        channel_nonzero_min=np.count_nonzero(future, axis=(2, 3)).min(0).tolist(),
        channel_nonzero_max=np.count_nonzero(future, axis=(2, 3)).max(0).tolist(),
        selected_frames=selected, tokens=tokens, time_seconds=times,
        coordinate='raw [LiDAR X, LiDAR Y]; display np.rot90(raw, 1); forward up/right right',
        reference='same-token annotation boxes transformed into the candidate LiDAR frame; standard bbox yaw',
        binary=bool(((layouts == 0) | (layouts == 1)).all()),
    )
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (output / 'README.md').write_text(
        '# BEV layout 可视化\n\n'
        f'- `bevlayout.gif`：{len(frames)} 帧，{args.fps:g} FPS；左为静态地图，中为地图与目标，右为目标及同 token 标注框黑色轮廓。\n'
        '- `keyframes.png`：当前帧及未来关键帧拼图。\n'
        '- `channels_frame_000001.png`：首个未来帧的 18 个独立通道。\n'
        '- `frame_*.png`：关键帧三栏高清图。\n'
        '- `metadata.json`：来源、SHA256、token、帧时间及通道统计。\n\n'
        '范围 80×80 m，分辨率 0.4 m。上为 LiDAR +Y（近似车头方向），右为 LiDAR +X；十字为 LiDAR 原点。'
        '只旋转显示，不修改 layout。静态通道为多标签叠加，后面的通道可能遮盖前面的通道。\n\n'
        '参考框先以同 token 的原始 2 Hz 标注校验格式。12 Hz 旧格式 `[w,l,h,-yaw-pi/2]` '
        '转换为标准 `[l,w,h,yaw]` 后绘制，避免把标注格式差异误判为后续帧错误。\n\n'
        f'第 0 帧来源：`{current_source}`；显式修复预览：{args.current_layout is not None}。未来 {len(future)} 帧来自保存的 `sequence.npz[bev_layout]`，'
        '与原 overview 同帧号、同 token。时间沿用 frames.csv 的 12 Hz 展示时间，并非重新计算采集时间。\n\n'
        '复现：\n\n```bash\n'
        f'conda run --no-capture-output -n loworld python tools/visualization/bev_layout.py --scene-dir {args.scene_dir}\n'
        '```\n')
    print(json.dumps({key: metadata[key] for key in
                      ('frames', 'fps', 'duration_ms', 'binary', 'source_sha256')}, indent=2))
    print(output)


if __name__ == '__main__':
    main()
