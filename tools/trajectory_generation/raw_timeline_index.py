"""Bounded-memory, reusable index of raw nuScenes temporal geometry."""
import json
from pathlib import Path
import sqlite3
import tempfile

from tools.genie.occ_generation.scene_runtime import file_identity, identity_key
from tools.trajectory_generation.resample_scene_timeline import iter_json_array


class RawTimelineIndex:
    """Scan raw JSON once, then retrieve physical scenes for arbitrary starts."""

    VERSION = 1

    def __init__(self, data_root, cache_root, version='v1.0-trainval'):
        database = Path(data_root) / version
        names = ('sample', 'sample_data', 'sample_annotation', 'ego_pose',
                 'instance', 'category', 'sensor', 'calibrated_sensor')
        self.paths = {name: database / f'{name}.json' for name in names}
        self.signature = {name: file_identity(path) for name, path in self.paths.items()}
        cache_root = Path(cache_root)
        cache_root.mkdir(parents=True, exist_ok=True)
        key = identity_key(dict(version=self.VERSION, sources=self.signature))
        self.path = cache_root / f'raw_timeline_{key}.sqlite'
        if not self.path.exists():
            self._build(cache_root)
        self.connection = sqlite3.connect(f'file:{self.path.resolve()}?mode=ro', uri=True)
        self.small = json.loads(self.connection.execute('SELECT payload FROM metadata').fetchone()[0])

    def _build(self, cache_root):
        with tempfile.NamedTemporaryFile(dir=cache_root, prefix='.raw_timeline_', delete=False) as stream:
            temporary = Path(stream.name)
        connection = sqlite3.connect(temporary)
        try:
            connection.executescript('''
                PRAGMA journal_mode=OFF;
                PRAGMA synchronous=OFF;
                PRAGMA temp_store=MEMORY;
                PRAGMA cache_size=-131072;
                CREATE TABLE sample(token TEXT PRIMARY KEY, scene TEXT, payload TEXT);
                CREATE TABLE sensor(scene TEXT, ego TEXT, payload TEXT);
                CREATE TABLE ego(token TEXT PRIMARY KEY, payload TEXT);
                CREATE TABLE annotation(scene TEXT, instance TEXT, payload TEXT);
                CREATE TABLE metadata(payload TEXT);
            ''')
            small = {name: {r['token']: r for r in iter_json_array(self.paths[name])}
                     for name in ('category', 'sensor', 'calibrated_sensor', 'instance')}
            connection.execute('INSERT INTO metadata VALUES(?)', (json.dumps(small),))
            sample_scene = {}

            def insert_rows(sql, rows):
                batch = []
                for row in rows:
                    batch.append(row)
                    if len(batch) == 4096:
                        connection.executemany(sql, batch)
                        batch.clear()
                if batch:
                    connection.executemany(sql, batch)
                connection.commit()

            print('Index raw timeline: samples', flush=True)
            def samples():
                for r in iter_json_array(self.paths['sample']):
                    sample_scene[r['token']] = r['scene_token']
                    payload = {k: r[k] for k in ('token', 'scene_token', 'timestamp')}
                    yield r['token'], r['scene_token'], json.dumps(payload)
            insert_rows('INSERT INTO sample VALUES(?,?,?)', samples())
            print('Index raw timeline: sensor acquisitions', flush=True)
            def sensors():
                for r in iter_json_array(self.paths['sample_data']):
                    calibration = small['calibrated_sensor'][r['calibrated_sensor_token']]
                    channel = small['sensor'][calibration['sensor_token']]['channel']
                    if channel != 'LIDAR_TOP' and not channel.startswith('CAM_'):
                        continue
                    payload = {k: r[k] for k in ('token', 'sample_token', 'timestamp',
                               'ego_pose_token', 'calibrated_sensor_token', 'filename')}
                    yield sample_scene[r['sample_token']], r['ego_pose_token'], json.dumps(payload)
            insert_rows('INSERT INTO sensor VALUES(?,?,?)', sensors())
            print('Index raw timeline: ego poses', flush=True)
            insert_rows('INSERT INTO ego VALUES(?,?)',
                        ((r['token'], json.dumps(r)) for r in iter_json_array(self.paths['ego_pose'])))
            print('Index raw timeline: instance annotations', flush=True)
            def annotations():
                for r in iter_json_array(self.paths['sample_annotation']):
                    payload = {k: r[k] for k in ('token', 'sample_token', 'instance_token',
                               'translation', 'size', 'rotation', 'num_lidar_pts', 'num_radar_pts')}
                    yield sample_scene[r['sample_token']], r['instance_token'], json.dumps(payload)
            insert_rows('INSERT INTO annotation VALUES(?,?,?)', annotations())
            connection.executescript('''
                CREATE INDEX sample_scene ON sample(scene);
                CREATE INDEX sensor_scene ON sensor(scene);
                CREATE INDEX annotation_scene ON annotation(scene);
            ''')
            connection.commit()
            connection.close()
            temporary.replace(self.path)
        finally:
            connection.close()
            temporary.unlink(missing_ok=True)

    def scene(self, target_token):
        row = self.connection.execute('SELECT scene FROM sample WHERE token=?', (target_token,)).fetchone()
        if row is None:
            raise ValueError(f'No native raw sample for target token {target_token}')
        scene = row[0]
        def records(table):
            return [json.loads(r[0]) for r in self.connection.execute(
                f'SELECT payload FROM {table} WHERE scene=?', (scene,))]
        ego = [json.loads(r[0]) for r in self.connection.execute('''
            SELECT DISTINCT ego.payload FROM sensor JOIN ego ON sensor.ego=ego.token
            WHERE sensor.scene=?''', (scene,))]
        annotations = records('annotation')
        instance_ids = {r['instance_token'] for r in annotations}
        return dict(signature=self.signature, target_token=target_token,
                    samples=records('sample'), sensor_data=records('sensor'),
                    ego={r['token']: r for r in ego}, annotations=annotations,
                    instances={key: self.small['instance'][key] for key in instance_ids},
                    **{key: self.small[key] for key in ('category', 'sensor', 'calibrated_sensor')})

    def close(self):
        self.connection.close()
