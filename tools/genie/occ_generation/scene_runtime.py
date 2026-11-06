"""Atomic scene artifacts, cache identities, and wall-clock stage timings."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import tempfile
import time

logger = logging.getLogger(__name__)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                inode=stat.st_ino)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def identity_key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


@contextmanager
def atomic_file(path, mode='wb'):
    """Replace only a completely written artifact; never reuse a shared temp name."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode=mode, dir=path.parent,
                                         prefix=f'.{path.name}.', delete=False) as stream:
            temporary = Path(stream.name)
            yield stream
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path, value):
    with atomic_file(path, 'w') as stream:
        json.dump(value, stream, indent=2, default=str)
        stream.write('\n')


class StageTimings:
    """Persist completed and failed stages, including synchronized GPU work."""

    def __init__(self, path, metadata=None):
        self.path = Path(path)
        self.started = time.perf_counter()
        self.data = dict(started_at=utc_now(), status='running', stages_seconds={},
                         **(metadata or {}))
        self.flush()

    def flush(self):
        self.data['wall_seconds'] = time.perf_counter() - self.started
        write_json(self.path, self.data)

    @contextmanager
    def stage(self, name, synchronize=None):
        if synchronize is not None:
            synchronize()
        start = time.perf_counter()
        logger.info('Starting %s', name)
        try:
            yield
        except BaseException:
            self.data['failed_stage'] = name
            raise
        finally:
            if synchronize is not None:
                synchronize()
            seconds = time.perf_counter() - start
            self.data['stages_seconds'][name] = seconds
            self.flush()
            logger.info('%s: %.3f s', name, seconds)

    def finish(self, error=None):
        self.data.update(status='failed' if error else 'completed', finished_at=utc_now())
        if error:
            self.data['error'] = str(error)
            self.data['error_type'] = type(error).__name__
        self.flush()
