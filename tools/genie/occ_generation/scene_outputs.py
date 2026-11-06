"""Scene deliverables, frame documentation and versioned run directories.

The entry point can import this module without loading numpy or model packages.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
import zipfile

from tools.genie.occ_generation.scene_runtime import (
    atomic_file, sha256_file, utc_now, write_json,
)

CLASSES = ('others', 'barrier', 'bicycle', 'bus', 'car',
           'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone',
           'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk',
           'terrain', 'manmade', 'vegetation', 'free')

FIELD_DESCRIPTIONS = {
    'target_token': 'Current frame token',
    'future_tokens_12hz': 'Future timestamp tokens from the original scene; index j maps to frame j+1; counterfactual trajectories do not imply ground truth at the candidate position',
    'current_occ': 'Reconstructed current OCC; X,Y,Z axis order matches the renderer',
    'pred_occ_12hz': 'Future OCC in T,X,Y,Z order; T excludes the current frame',
    'current_points_xyz': 'Reconstructed current point cloud in LiDAR coordinates, in meters',
    'pred_points_xyz_concat': 'Future point clouds concatenated in frame order, in LiDAR coordinates and meters',
    'pred_points_xyz_offsets': 'Future frame j uses concat[offsets[j]:offsets[j+1]]',
    'point_coordinate': 'Coordinate system of the saved points',
    'point_lidar2ego': 'Current LiDAR-to-ego homogeneous transform; column-vector convention',
    'pred_points_lidar2ego': 'LiDAR-to-ego homogeneous transform for each future frame; column-vector convention',
    'bev_layout': 'Future BEV layout conditions, T,C,H,W',
    'current_bev_layout': 'Current BEV layout actually consumed by the model, C,H,W; physical box orientations are consistent with future conditions',
    'candidate_curr2futu_ego': 'Current-ego-to-future-ego homogeneous transforms; column-vector convention',
    'scene_name': 'Scene name in the 12 Hz annotations',
    'generation_seed': 'Random seed for generation',
    'trajectory_name': 'Selected trajectory name in the counterfactual condition file',
    'trajectory_index': 'Candidate index in the counterfactual condition file',
    'trajectory_source_sha256': 'SHA256 of the counterfactual trajectory and layout condition file',
}


def relative_path(path, root):
    return Path(os.path.relpath(Path(path).resolve(), Path(root).resolve())).as_posix()


def resolve_prediction(root):
    """Resolve a completed prediction from its scene or run manifest."""
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['kind'] == 'scene_collection':
        latest = manifest.get('latest_successful_run')
        if not latest:
            raise FileNotFoundError(f'No successful scene predictions found in {root}')
        return resolve_prediction(root / latest)
    if manifest['kind'] != 'scene_run' or manifest['status'] != 'completed':
        raise ValueError(f'Run is not a completed prediction source: {root}')
    prediction = (root / manifest['prediction']['path']).resolve()
    if not prediction.is_file():
        raise FileNotFoundError(prediction)
    if sha256_file(prediction) != manifest['prediction']['sha256']:
        raise ValueError(f'Prediction no longer matches its manifest: {prediction}')
    return prediction


def run_logged_process(command, *, cwd, log_path, env=None):
    """Keep complete worker output on disk while streaming it to the terminal."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w', buffering=1) as log:
        log.write('Command: ' + json.dumps(command) + '\n\n')
        with subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True,
                              encoding='utf-8', errors='replace', bufsize=1) as process:
            try:
                for line in process.stdout:
                    log.write(line)
                    sys.stdout.write(line)
                    sys.stdout.flush()
                code = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
    if code:
        raise subprocess.CalledProcessError(code, command)


class SceneBundle:
    """One immutable result directory per invocation; caches belong to the scene."""

    def __init__(self, root, scene, mode, parameters):
        self.root = Path(root).resolve()
        self.scene, self.mode, self.parameters = scene, mode, parameters
        self.run_id = (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                       + '_' + uuid.uuid4().hex[:8])
        self.path = self.root / 'runs' / self.run_id
        self.cache = self.root / '.cache'
        self.data = dict(schema_version=2, kind='scene_run', run_id=self.run_id,
                         scene=scene, mode=mode, started_at=utc_now(),
                         status='running', parameters=parameters, artifacts={})

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / '.scene.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise RuntimeError(f'Another scene run is using {self.root}') from None
        try:
            index_path = self.root / 'manifest.json'
            self.index = json.loads(index_path.read_text()) if index_path.exists() else dict(
                schema_version=2, kind='scene_collection', scene=self.scene, runs=[])
            if self.index.get('kind') != 'scene_collection' or self.index.get('scene') != self.scene:
                raise ValueError(f'Output directory belongs to a different scene or format: {self.root}')
            for name in ('metadata', 'logs', 'predictions', 'visualization'):
                (self.path / name).mkdir(parents=True, exist_ok=False)
            self.index['runs'].append(dict(run_id=self.run_id,
                                          path=relative_path(self.path, self.root),
                                          mode=self.mode, status='running'))
            self.publish()
            return self
        except BaseException:
            self.lock.close()
            raise

    def publish(self):
        write_json(self.path / 'manifest.json', self.data)
        self.index['runs'][-1]['status'] = self.data['status']
        self.index['latest_run'] = relative_path(self.path, self.root)
        write_json(self.root / 'manifest.json', self.index)
        self.write_readme()
        rows = ['# Scene generation results', '', f'Scene: `{self.scene}`. Each run is saved separately under `runs/`.', '',
                '`latest/` points to the most recent successful run. Check the manifest for the status of the latest attempt.', '',
                '| Run | Mode | Status |', '| --- | --- | --- |']
        for item in reversed(self.index['runs']):
            rows.append(f"| [{item['run_id']}]({item['path']}/README.md) | {item['mode']} | {item['status']} |")
        if (self.root / 'predictions.npz').exists():
            rows.extend(['', 'Legacy outputs at the directory root are preserved and are not overwritten by new runs.'])
        with atomic_file(self.root / 'README.md', 'w') as stream:
            stream.write('\n'.join(rows) + '\n')

    def write_readme(self):
        data = self.data
        lines = ['# OCC and LiDAR scene results', '',
                 f"Scene: `{self.scene}`; run: `{self.run_id}`; status: **{data['status']}**.", '',
                 f"Mode: `{self.mode}`; started at (UTC): {data['started_at']}.", '',
                 '## View results', '']
        if data.get('error'):
            lines.extend([f"Run failed: {data['error'].splitlines()[0]}", ''])
        for name, item in data['artifacts'].items():
            lines.append(f"- [{name}]({item['path']}): {item['description']}")
        prediction = data.get('prediction')
        if prediction:
            lines.extend(['', f"Complete data: [sequence.npz]({prediction['path']}).", '',
                          f"SHA256: `{prediction['sha256']}`."])
            if self.mode == 'render_only':
                original = data['prediction_source']
                lines.extend(['', 'This run rerenders an existing prediction without running the model again.', '',
                              f"Original prediction: `{original['path']}`. This directory includes the complete NPZ and can be shared independently.", '',
                              f"Storage: `{original['storage']}` (prefer a hard link on the same filesystem; copy across filesystems; prediction contents are preserved)."])
        summary = data.get('sequence', {})
        if summary:
            lines.extend(['', '## Frame and coordinate conventions', '',
                          f"{summary['frames']} frames: current reconstruction + {summary['future_frames']} future predictions, sampled at 12 Hz.", '',
                          '`frame_000000` is the current reconstruction; `frame_000001` is the first future frame (+1/12 s).',
                          'Future array index = frame number - 1; CSV point_start is inclusive and point_stop is exclusive.',
                          'Each point cloud uses its own frame LiDAR coordinates, in meters; apply the corresponding lidar2ego transform to obtain ego coordinates.',
                          'OCC uses an ego grid spanning [-40,40) × [-40,40) × [-1,5.4) meters, with 0.4-meter voxels.',
                          'Voxel center = [-40,-40,-1] + (index + 0.5) × 0.4.', '',
                          f"Recorded seed: `{summary.get('seed', 'not recorded')}`; scene: `{summary.get('scene', 'not recorded')}`.", '',
                          '## NPZ fields', '', '| Field | shape | dtype | Description |', '| --- | --- | --- | --- |'])
            for key, spec in data['arrays'].items():
                lines.append(f"| `{key}` | `{spec['shape']}` | `{spec['dtype']}` | {FIELD_DESCRIPTIONS.get(key, 'Additional field')} |")
            lines.extend(['', '## OCC classes', '',
                          '; '.join(f'{i}: {name}' for i, name in enumerate(CLASSES)), '',
                          '`17=free`; `255=ignore` (if present). OCC colors use the repository OCC3D palette; LiDAR colors are mapped by height.'])
        source = data.get('generation', {})
        if source:
            lines.extend(['', '## Model source', '',
                          f"Config: `{source.get('config')}`", '',
                          f"Checkpoint: `{source.get('checkpoint')}`", '',
                          f"Seed: `{source.get('seed')}`; checkpoint SHA256: `{source.get('checkpoint_sha256', 'not recorded')}`."])
            if source.get('trajectory'):
                trajectory = source['trajectory']
                lines.extend(['', f"Trajectory: `{trajectory['name']}` (candidate {trajectory['index']}); "
                              f"condition SHA256: `{trajectory['source']['sha256']}`."])
            if (self.path / 'metadata' / 'inference.json').exists():
                lines.extend(['', 'See [inference.json](metadata/inference.json) for complete provenance and inference stage timings.'])
        elif self.mode == 'render_only':
            lines.extend(['', 'The original file provides no verifiable model provenance; the current CLI defaults for checkpoint and seed do not describe the original prediction.'])
        lines.extend(['', '## File guide', '',
                      '- `predictions/`: complete predictions, frame index and array schema.',
                      '- `visualization/`: GIF, optional PNGs and rendering parameters.',
                      '- `exports/lidar/`: optional point clouds for each frame, including the current reconstruction and future frames.',
                      '- `metadata/`: configuration, provenance, status and stage timings.',
                      '- `logs/`: complete subprocess output and MMEngine logs.',
                      '- `.cache/` at the scene root: regenerable input caches; deleting them does not affect saved predictions.'])
        with atomic_file(self.path / 'README.md', 'w') as stream:
            stream.write('\n'.join(lines) + '\n')

    def artifact(self, name, path, description):
        path = Path(path)
        self.data['artifacts'][name] = dict(path=relative_path(path, self.path),
                                           description=description)

    def attach_prediction(self, source):
        """Keep rerenders independently shareable without duplicate disk blocks."""
        source = Path(source).resolve()
        digest = sha256_file(source)
        destination = self.path / 'predictions' / 'sequence.npz'
        # The prediction refers to its frozen clock relative to itself. Keep
        # that input alongside a rerender so semantic export stays portable.
        from tools.genie.semantic_export.scene_inputs import prediction_annotation
        annotation = prediction_annotation(source)
        import numpy as np
        with np.load(source, allow_pickle=False) as archive:
            relative_annotation = str(archive['timeline_annotation'])
            expected = str(archive['timeline_annotation_sha256'])
        snapshot = (destination.parent / relative_annotation).resolve()
        if not snapshot.is_relative_to(self.path.resolve()):
            raise ValueError('Frozen annotation must remain inside the result bundle')
        with annotation.open('rb') as original, atomic_file(snapshot) as stream:
            shutil.copyfileobj(original, stream)
        if sha256_file(snapshot) != expected:
            raise ValueError('Frozen prediction timeline changed while copying')
        try:
            os.link(source, destination)
            storage = 'hardlink'
        except OSError:
            with source.open('rb') as original, atomic_file(destination) as stream:
                shutil.copyfileobj(original, stream)
            storage = 'copy'
        self.data['prediction_source'] = dict(path=str(source), sha256=digest, storage=storage)
        source_manifest = source.parent.parent / 'manifest.json'
        if source.parent.name == 'predictions' and source_manifest.exists():
            origin = json.loads(source_manifest.read_text())
            if origin.get('prediction', {}).get('sha256') == digest:
                self.data['origin_run_id'] = origin.get('origin_run_id') or origin.get('run_id')
                if origin.get('generation'):
                    self.data['generation'] = origin['generation']
        return destination

    def describe_prediction(self, prediction, *, export_points=False):
        # Only the small temporal arrays are decompressed for indexing. Read NPY
        # headers from the ZIP members rather than loading voxel/point arrays.
        import numpy as np
        prediction = Path(prediction).resolve()
        arrays = {}
        with zipfile.ZipFile(prediction) as archive:
            for member in archive.namelist():
                if not member.endswith('.npy'):
                    continue
                with archive.open(member) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version not in ((1, 0), (2, 0)):
                        raise ValueError(f'Unsupported NPY header version: {version}')
                    reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                              else np.lib.format.read_array_header_2_0)
                    shape, _, dtype = reader(stream)
                arrays[member[:-4]] = dict(shape=list(shape), dtype=str(dtype))
        with np.load(prediction, allow_pickle=False) as archive:
            tokens = [str(archive['target_token'].item()),
                      *map(str, archive['future_tokens_12hz'])]
            offsets = archive['pred_points_xyz_offsets']
            current_count = arrays['current_points_xyz']['shape'][0]
            if str(archive['point_coordinate'].item()) != 'lidar':
                raise ValueError('Expected points in LiDAR coordinates')
            if arrays['current_occ']['shape'] != [200, 200, 16]:
                raise ValueError('Expected the nuScenes OCC3D 200x200x16 ego grid')
            summary = dict(frames=len(tokens), future_frames=len(tokens) - 1, fps=12)
            for field, key in [('generation_seed', 'seed'), ('scene_name', 'scene')]:
                if field in archive:
                    summary[key] = archive[field].item()
            if summary.get('scene', self.scene) != self.scene:
                raise ValueError(f"Prediction scene {summary['scene']} does not match {self.scene}")
            if len(offsets) != len(tokens) or arrays['pred_occ_12hz']['shape'][0] != len(tokens) - 1:
                raise ValueError('Prediction temporal arrays have inconsistent lengths')
            if (offsets.ndim != 1 or offsets.dtype.kind not in 'iu' or offsets[0] != 0
                    or np.any(offsets[1:] < offsets[:-1])
                    or offsets[-1] != arrays['pred_points_xyz_concat']['shape'][0]):
                raise ValueError('Prediction point offsets are invalid')
            if arrays['pred_points_lidar2ego']['shape'] != [len(tokens) - 1, 4, 4]:
                raise ValueError('Future point transforms do not match the frame count')
            self.data.update(sequence=summary, arrays=arrays,
                             prediction=dict(path=relative_path(prediction, self.path),
                                             sha256=sha256_file(prediction)))
            original = self.data.get('prediction_source')
            if original and original['sha256'] != self.data['prediction']['sha256']:
                raise RuntimeError('Prediction source changed while preparing this run')
            index_path = self.path / 'predictions' / 'frames.csv'
            with atomic_file(index_path, 'w') as stream:
                writer = csv.writer(stream, lineterminator='\n')
                writer.writerow(['frame_id', 'kind', 'token', 'relative_time_seconds',
                                 'occ_key', 'occ_index', 'points_key', 'point_start',
                                 'point_stop', 'lidar2ego_key', 'lidar2ego_index', 'export_path'])
                for i, token in enumerate(tokens):
                    frame_id = f'frame_{i:06d}'
                    output = self.path / 'exports' / 'lidar' / f'{frame_id}.npz'
                    if i == 0:
                        occ, occ_index = 'current_occ', ''
                        pts, start, stop = 'current_points_xyz', 0, current_count
                        transform, transform_index = 'point_lidar2ego', ''
                    else:
                        occ, occ_index = 'pred_occ_12hz', i - 1
                        pts, start, stop = 'pred_points_xyz_concat', int(offsets[i - 1]), int(offsets[i])
                        transform, transform_index = 'pred_points_lidar2ego', i - 1
                    writer.writerow([frame_id, 'current_reconstruction' if i == 0 else 'future_prediction',
                                     token, f'{i / 12:.9f}', occ, occ_index, pts, start, stop,
                                     transform, transform_index,
                                     relative_path(output, self.path) if export_points else ''])
            if export_points:
                # Explicit export decompresses each packed array once.
                current = archive['current_points_xyz']
                future = archive['pred_points_xyz_concat']
                transforms = archive['pred_points_lidar2ego']
                for i, token in enumerate(tokens):
                    points = current if i == 0 else future[offsets[i - 1]:offsets[i]]
                    transform = archive['point_lidar2ego'] if i == 0 else transforms[i - 1]
                    output = self.path / 'exports' / 'lidar' / f'frame_{i:06d}.npz'
                    with atomic_file(output) as stream:
                        np.savez_compressed(stream, points=points, frame_index=np.asarray(i),
                                            token=np.asarray(token), relative_time_seconds=np.asarray(i / 12),
                                            coordinate=np.asarray('lidar'), lidar2ego=transform)
                self.artifact('Point clouds by frame', self.path / 'exports' / 'lidar', 'Frame numbers, points, tokens, times and extrinsics')
        write_json(self.path / 'predictions' / 'schema.json', dict(
            schema_version=2, arrays=arrays, descriptions=FIELD_DESCRIPTIONS,
            fps=12, point_coordinate='lidar', units='meters', occ_axis_order='X,Y,Z',
            occ_origin_m=[-40, -40, -1], voxel_size_m=0.4,
            classes=dict(enumerate(CLASSES)), prediction=self.data['prediction']))
        self.artifact('Frame index', index_path, 'Frame numbers, tokens, relative times and array slices')
        self.artifact('Data schema', self.path / 'predictions' / 'schema.json', 'Fields, dimensions, classes and coordinate conventions')
        self.publish()

    def __exit__(self, kind, error, traceback):
        try:
            self.data.update(status='failed' if kind else 'completed', finished_at=utc_now())
            if kind:
                self.data.update(error=str(error), error_type=kind.__name__)
            else:
                # Only successful runs can replace the convenient latest view.
                temporary = self.root / f'.latest_{self.run_id}'
                try:
                    temporary.symlink_to(relative_path(self.path, self.root), target_is_directory=True)
                    temporary.replace(self.root / 'latest')
                finally:
                    temporary.unlink(missing_ok=True)
                self.index['latest_successful_run'] = relative_path(self.path, self.root)
            self.publish()
        finally:
            self.lock.close()
