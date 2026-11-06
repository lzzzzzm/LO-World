"""Line-based progress for concurrent FVD evaluation workers."""

import logging
import threading
import time
from contextlib import contextmanager


class EvaluationProgress:
    def __init__(self, label, heartbeat_seconds=30):
        self.logger = logging.getLogger(f"wcoda_fvd.{label}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                f"%(asctime)s | %(levelname)s | {label} | %(message)s",
                datefmt="%H:%M:%S"))
            self.logger.addHandler(handler)
        self.heartbeat_seconds = heartbeat_seconds
        self._lock = threading.RLock()
        self._last_message = time.monotonic()

    def info(self, message, *args):
        with self._lock:
            self._last_message = time.monotonic()
            self.logger.info(message, *args)

    @contextmanager
    def stage(self, name):
        started = time.monotonic()
        stopped = threading.Event()
        self.info("Starting: %s", name)

        def heartbeat():
            while not stopped.wait(self.heartbeat_seconds):
                with self._lock:
                    if time.monotonic() - self._last_message >= self.heartbeat_seconds:
                        self.info("Still running: %s (elapsed %.1fs)",
                                  name, time.monotonic() - started)

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            yield
        except BaseException:
            self.info("Failed: %s (elapsed %.1fs)", name, time.monotonic() - started)
            raise
        else:
            self.info("Completed: %s (elapsed %.1fs)", name, time.monotonic() - started)
        finally:
            stopped.set()
            thread.join()
