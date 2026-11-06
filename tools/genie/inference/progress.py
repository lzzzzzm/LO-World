"""Wall-clock phase logs and periodic updates, with no model/GPU operations."""

from contextlib import contextmanager
import logging
from pathlib import Path
import sys
import threading
import time

from tools.genie.occ_generation.scene_runtime import utc_now, write_json


class InferenceProgress:
    def __init__(self, heartbeat_seconds=30):
        self.logger = logging.getLogger('genie_video')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S'))
            self.logger.addHandler(handler)
        self.interval = heartbeat_seconds
        self.path = None
        self.lock = threading.RLock()
        self.started = time.perf_counter()
        self.last_reported = self.started
        self.stage_started = self.started
        self.detail_started = self.started
        self.data = dict(status='running', started_at=utc_now(), stages=[])

    def info(self, message, *args):
        with self.lock:
            self.logger.info(message, *args)
            self.last_reported = time.perf_counter()

    def set_output(self, path, **metadata):
        with self.lock:
            self.path = Path(path)
            self.data.update(metadata)
            self._flush()

    def _flush(self):
        self.data['updated_at'] = utc_now()
        self.data['wall_seconds'] = time.perf_counter() - self.started
        if self.path:
            write_json(self.path, self.data)

    def _close_detail(self):
        if self.data.get('detail'):
            elapsed = time.perf_counter() - self.detail_started
            self.data['subphases'].append(dict(name=self.data['detail'], seconds=elapsed))
            self.info('Substage finished: %s (%.1f s)', self.data['detail'], elapsed)

    def detail(self, message):
        with self.lock:
            self._close_detail()
            self.data['detail'] = message
            self.data.pop('denoising', None)
            self.detail_started = time.perf_counter()
            self.info('Running: %s', message)
            self._flush()

    def denoising_step(self, done, total):
        with self.lock:
            elapsed = time.perf_counter() - self.detail_started
            remaining = elapsed / done * (total - done)
            self.data['denoising'] = dict(step=done, total=total, elapsed_seconds=elapsed,
                                         estimated_remaining_seconds=remaining)
            if done == 1 or done % 5 == 0 or done == total:
                self.info('Denoising: %d/%d (%.0f%%), elapsed %.1f s, ETA %.1f s',
                          done, total, done * 100 / total, elapsed, remaining)
                self._flush()

    def _heartbeat(self, stop):
        while True:
            with self.lock:
                delay = max(0.1, self.interval - (time.perf_counter() - self.last_reported))
            if stop.wait(delay):
                return
            with self.lock:
                # A step or phase update may have arrived while waiting. Report
                # only after a full quiet interval, so we don't duplicate it.
                if time.perf_counter() - self.last_reported < self.interval:
                    continue
                detail = self.data.get('detail')
                elapsed = time.perf_counter() - (self.detail_started if detail else self.stage_started)
                denoising = self.data.get('denoising')
                suffix = (f"; {denoising['step']}/{denoising['total']} steps completed"
                          if denoising else '')
                self.info('Still running: %s (elapsed %.1f s%s)',
                          detail or self.data['current_stage'], elapsed, suffix)
                self._flush()

    @contextmanager
    def stage(self, name):
        with self.lock:
            self.stage_started = time.perf_counter()
            self.data.update(current_stage=name, detail=None, subphases=[])
            self.data.pop('denoising', None)
            self.info('Stage started: %s', name)
            self._flush()
        stop = threading.Event()
        thread = threading.Thread(target=self._heartbeat, args=(stop,), daemon=True)
        thread.start()
        failed = False
        try:
            yield
        except BaseException as error:
            failed = True
            self.logger.error('Stage failed: %s: %s', name, error)
            raise
        finally:
            stop.set()
            thread.join()
            with self.lock:
                self._close_detail()
                seconds = time.perf_counter() - self.stage_started
                self.data['stages'].append(dict(name=name, seconds=seconds,
                                                status='failed' if failed else 'completed',
                                                subphases=self.data.pop('subphases')))
                self.data.update(current_stage=None, detail=None)
                self.info('Stage %s: %s (%.1f s)', 'failed' if failed else 'completed', name, seconds)
                self._flush()

    def finish(self, error=None):
        with self.lock:
            self.data.update(status='failed' if error is not None else 'completed', finished_at=utc_now())
            if error is not None:
                self.data.update(error=str(error), error_type=type(error).__name__)
            self._flush()
            self.info('Video inference %s, total elapsed %.1f s',
                      'failed' if error is not None else 'completed', self.data['wall_seconds'])
