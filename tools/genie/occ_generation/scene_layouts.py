"""Native-box geometry and verified context layouts for scene generation.

The shared renderer retains its historical local-Y reflection. Native nuScenes
boxes must therefore be converted to its legacy box convention explicitly.
Precomputed arrays without this provenance cannot establish correct geometry.
"""

import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

from tools.genie.occ_generation.scene_runtime import (
    atomic_file, file_identity, identity_key, sha256_file, write_json,
)
from tools.trajectory_generation.bev_layout import keyframe_info_to_render_info


CONTEXT_LAYOUT_CACHE_NAME = 'bev_layout_native_geometry_v1'
BOX_CONVENTION = 'native_lwh_yaw_to_legacy_wlh_negative_yaw_minus_pi_over_2_v1'


def native_to_renderer_info(info, scene_locations, class_names):
    """Copy standard native boxes into the renderer's legacy convention."""
    result = keyframe_info_to_render_info(info, scene_locations, class_names)
    boxes = result['gt_boxes'].copy()
    boxes[:, [3, 4]] = boxes[:, [4, 3]]
    boxes[:, 6] = -boxes[:, 6] - np.pi / 2
    return dict(result, gt_boxes=boxes, box_convention='legacy')


def ensure_native_layouts(renderer, native_infos, class_names, tokens,
                          cache_root, scene_locations, data_root):
    """Reuse only arrays with matching geometry, map, code and file digest.

    Never copy token-only precomputed layouts: native and interpolated infos
    share token names but use different box conventions. Old scene caches lack
    a verified geometry identity and are deliberately rebuilt.
    """
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    by_token = {str(info['token']): info for info in native_infos}
    implementation = {
        str(path): sha256_file(path) for path in (
            Path(__file__).resolve(),
            Path(__file__).resolve().parents[2] / 'trajectory_generation/bev_layout.py',
            Path(__file__).resolve().parents[2] / 'trajectory_generation/box_conventions.py',
            Path(__file__).resolve().parents[2] / 'trajectory_generation/map_geometry_cache.py',
        )
    }
    statistics = dict(rendered=0, hits=0)
    for token in tokens:
        token = str(token)
        if token not in by_token:
            raise KeyError(f'No native nuScenes info for BEVLayout token {token}')
        render_info = native_to_renderer_info(
            by_token[token], scene_locations, class_names)
        map_path = Path(data_root) / 'maps/expansion' / f"{render_info['location']}.json"
        signature = identity_key(dict(
            version=1, convention=BOX_CONVENTION,
            frame_sha256=hashlib.sha256(pickle.dumps(render_info, protocol=4)).hexdigest(),
            classes=list(class_names), map=file_identity(map_path),
            geometry=dict(shape=[18, renderer.canvas_h, renderer.canvas_w],
                          lidar2canvas=renderer.lidar2canvas.tolist()),
            implementation=implementation,
        ))
        destination = cache_root / f'{token}.npy'
        sidecar = cache_root / f'{token}.json'
        try:
            metadata = json.loads(sidecar.read_text())
            if (metadata['signature'] == signature
                    and metadata['sha256'] == sha256_file(destination)):
                values = np.load(destination, allow_pickle=False)
                if (values.shape == (18, renderer.canvas_h, renderer.canvas_w)
                        and values.dtype == np.uint8
                        and ((values == 0) | (values == 1)).all()):
                    statistics['hits'] += 1
                    continue
        except (OSError, ValueError, KeyError, EOFError):
            pass
        values = renderer.render(render_info).astype(np.uint8)
        if (values.shape != (18, renderer.canvas_h, renderer.canvas_w)
                or not ((values == 0) | (values == 1)).all()):
            raise ValueError(f'Invalid rendered native layout: {token}')
        with atomic_file(destination) as stream:
            np.save(stream, values, allow_pickle=False)
        write_json(sidecar, dict(signature=signature, convention=BOX_CONVENTION,
                                sha256=sha256_file(destination), token=token))
        statistics['rendered'] += 1
    return statistics
