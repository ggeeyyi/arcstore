"""Training-log write-back: tee stdout/stderr to a local file + S3.

Two forms:

* console script ``arcstore-tee`` (this module's :func:`main`) — pipe a
  process's output through it from a shell::

      exec > >(arcstore-tee "s3://bucket/run/logs/host.log" \
               --local /tmp/run.log --interval 15) 2>&1

  stdin is passed through to stdout, appended to the local file, and the
  file is re-uploaded via the S3 API every ``--interval`` seconds while it
  keeps growing (loss window <= interval on a hard pod kill).

* in-process :class:`LogTee` — ``LogTee(local, s3_uri).install()`` wraps
  ``sys.stdout`` / ``sys.stderr``; ``close()`` does a final synchronous
  upload. The background flusher is a daemon thread.

Uploads always go through the S3 API (mountpoint-s3 rejects overwrites).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time

from .location import is_s3
from .uploads import upload_file

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_S = 15.0


class _Flusher:
    """Daemon thread re-uploading ``local_path`` to ``s3_uri`` while it grows."""

    def __init__(self, local_path: str, s3_uri: str, interval_s: float):
        self.local_path = local_path
        self.s3_uri = s3_uri
        self.interval_s = max(1.0, float(interval_s))
        self._stop = threading.Event()
        self._uploaded_size = -1
        self._thread = threading.Thread(
            target=self._run, name="arcstore-logtee", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def _upload_if_dirty(self) -> None:
        try:
            size = os.path.getsize(self.local_path)
        except OSError:
            return
        if size == self._uploaded_size:
            return
        try:
            upload_file(self.local_path, self.s3_uri)
            self._uploaded_size = size
        except Exception as e:  # noqa: BLE001 — log loss must never kill the run
            logger.warning(f"[arcstore-tee] upload failed ({e}); will retry.")

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._upload_if_dirty()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self.interval_s + 5.0)
        self._upload_if_dirty()  # final synchronous flush


class _TeeStream:
    """File-like wrapper writing to the original stream + the log file."""

    def __init__(self, orig, logfile):
        self._orig = orig
        self._logfile = logfile

    def write(self, s) -> int:
        n = self._orig.write(s)
        try:
            self._logfile.write(s)
            self._logfile.flush()
        except (OSError, ValueError):
            pass
        return n

    def flush(self) -> None:
        self._orig.flush()

    def __getattr__(self, name):
        return getattr(self._orig, name)


class LogTee:
    """In-process stdout/stderr tee with periodic S3 write-back."""

    def __init__(
        self,
        local_path: str,
        s3_uri: str,
        *,
        interval_s: float = DEFAULT_INTERVAL_S,
    ):
        if not is_s3(s3_uri):
            raise ValueError(f"LogTee expects an s3:// destination, got {s3_uri!r}")
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        self.local_path = local_path
        self.s3_uri = s3_uri
        self._file = open(local_path, "a", encoding="utf-8")
        self._flusher = _Flusher(local_path, s3_uri, interval_s)
        self._orig_stdout = None
        self._orig_stderr = None

    def install(self) -> "LogTee":
        self._orig_stdout, self._orig_stderr = sys.stdout, sys.stderr
        sys.stdout = _TeeStream(sys.stdout, self._file)
        sys.stderr = _TeeStream(sys.stderr, self._file)
        self._flusher.start()
        return self

    def close(self) -> None:
        if self._orig_stdout is not None:
            sys.stdout = self._orig_stdout
            sys.stderr = self._orig_stderr
            self._orig_stdout = self._orig_stderr = None
        try:
            self._file.flush()
        except ValueError:
            pass
        self._flusher.close()
        self._file.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="arcstore-tee",
        description="Tee stdin to stdout + a local file, with periodic S3 upload.",
    )
    parser.add_argument("s3_uri", help="s3://bucket/key destination for the log file")
    parser.add_argument(
        "--local",
        default=None,
        help="local log path (default: /tmp/arcstore-tee/<basename of s3 key>)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL_S,
        help=f"seconds between uploads (default {DEFAULT_INTERVAL_S:g})",
    )
    args = parser.parse_args(argv)

    if not is_s3(args.s3_uri):
        parser.error(f"destination must be an s3:// URI, got {args.s3_uri!r}")
    local = args.local or os.path.join(
        "/tmp/arcstore-tee", os.path.basename(args.s3_uri.rstrip("/")) or "run.log"
    )
    os.makedirs(os.path.dirname(local) or ".", exist_ok=True)

    flusher = _Flusher(local, args.s3_uri, args.interval)
    flusher.start()
    stdout = sys.stdout.buffer
    try:
        with open(local, "ab") as f:
            for line in iter(sys.stdin.buffer.readline, b""):
                stdout.write(line)
                stdout.flush()
                f.write(line)
                f.flush()
    finally:
        flusher.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
