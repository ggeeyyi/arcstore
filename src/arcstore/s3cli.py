"""s5cmd / aws-CLI subprocess layer with boto3 fallback.

Tool preference everywhere: ``s5cmd`` (multi-threaded, ~1 GiB/s) -> ``aws``
CLI -> boto3. Callers never shell out themselves; they use :func:`ls_prefix`
and :func:`download_file` here, or the bulk helpers in
:mod:`arcstore.uploads`.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Sequence

from .location import split_s3

logger = logging.getLogger(__name__)

_NOT_FOUND_MARKERS = (
    "nosuchkey",
    "404",
    "not found",
    "does not exist",
    "key not found",
    "no such file or directory",
    "no object found",
)


def have_s5cmd() -> bool:
    return shutil.which("s5cmd") is not None


def have_aws() -> bool:
    return shutil.which("aws") is not None


def _looks_not_found(output: str) -> bool:
    low = output.lower()
    return any(m in low for m in _NOT_FOUND_MARKERS)


@dataclass(frozen=True)
class LsEntry:
    """One immediate child of an S3 prefix listing."""

    name: str  # basename; directories carry a trailing "/"
    size: int | None  # None for directories
    is_dir: bool


@dataclass(frozen=True)
class CliRunResult:
    """Successful S3 CLI invocation."""

    tool: str
    stdout: str | bytes
    stderr: str | bytes


def run_cli_candidates(
    candidates: Sequence[tuple[str, list[str]]],
    *,
    text: bool = True,
) -> tuple[CliRunResult | None, str | None, str | None]:
    """Try available CLI backends in order.

    Returns ``(result, last_error, not_found_error)``. Exactly one of
    ``result`` or ``not_found_error`` is set on success / missing-object.
    ``last_error`` is the most recent non-missing backend failure.
    """
    last_err: str | None = None
    for tool, cmd in candidates:
        if shutil.which(tool) is None:
            continue
        try:
            proc = subprocess.run(cmd, capture_output=True, text=text, check=False)
        except OSError as e:
            last_err = f"{tool} failed to start: {e}"
            continue
        stdout = proc.stdout or ("" if text else b"")
        stderr = proc.stderr or ("" if text else b"")
        if proc.returncode == 0:
            return CliRunResult(tool=tool, stdout=stdout, stderr=stderr), None, None
        if text:
            combined = str(stderr) + str(stdout)
        else:
            combined = bytes(stderr).decode("utf-8", "replace") + bytes(stdout).decode(
                "utf-8", "replace"
            )
        if _looks_not_found(combined):
            return None, last_err, f"{tool}: {combined.strip()[:400]}"
        last_err = f"{tool} rc={proc.returncode}: {combined.strip()[:400]}"
    return None, last_err, None


def _parse_cli_ls(stdout: str, tool: str) -> list[LsEntry]:
    """Parse ``s5cmd ls`` / ``aws s3 ls`` output into entries.

    s5cmd rows: ``DIR  name/`` or ``2026/01/01 00:00:00  <size>  name``.
    aws rows:   ``PRE name/``  or ``2026-01-01 00:00:00  <size> name``.
    """
    entries: list[LsEntry] = []
    for line in stdout.splitlines():
        cols = line.split()
        if not cols:
            continue
        name = cols[-1]
        is_dir = name.endswith("/") or cols[0] in ("DIR", "PRE")
        size: int | None = None
        if not is_dir:
            for tok in cols:
                if tok.isdigit():
                    size = int(tok)
                    break
        if is_dir and not name.endswith("/"):
            name += "/"
        entries.append(LsEntry(name=name, size=size, is_dir=is_dir))
    return entries


def ls_prefix(s3_prefix: str) -> list[LsEntry]:
    """List the immediate children of an ``s3://`` prefix.

    Returns ``[]`` for an empty / nonexistent prefix ("no object found" is
    the expected first-run case, not an error). Raises ``RuntimeError`` only
    when every backend genuinely failed.
    """
    uri = s3_prefix.rstrip("/") + "/"

    result, last_err, not_found = run_cli_candidates(
        (
            ("s5cmd", ["s5cmd", "ls", uri]),
            ("aws", ["aws", "s3", "ls", uri]),
        )
    )
    if result is not None:
        return _parse_cli_ls(str(result.stdout), result.tool)
    if not_found is not None:
        return []

    try:
        return _boto3_ls_prefix(uri)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"[arcstore] failed to list {s3_prefix}: {last_err or e}"
        ) from e


def _boto3_ls_prefix(uri: str) -> list[LsEntry]:
    import boto3

    bucket, key = split_s3(uri)
    client = boto3.client("s3")
    paginator = client.get_paginator("list_objects_v2")
    entries: list[LsEntry] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=key, Delimiter="/"):
        for pre in page.get("CommonPrefixes", []):
            name = pre["Prefix"][len(key):]
            entries.append(LsEntry(name=name, size=None, is_dir=True))
        for obj in page.get("Contents", []):
            name = obj["Key"][len(key):]
            if not name:
                continue
            entries.append(LsEntry(name=name, size=obj["Size"], is_dir=False))
    return entries


def head_object(s3_uri: str) -> int | None:
    """Size of the object in bytes, or None if it does not exist."""
    if have_s5cmd() or have_aws():
        try:
            for e in ls_prefix_exact(s3_uri):
                if not e.is_dir:
                    return e.size if e.size is not None else 0
            return None
        except RuntimeError:
            pass
    try:
        import boto3
        from botocore.exceptions import ClientError

        bucket, key = split_s3(s3_uri)
        try:
            resp = boto3.client("s3").head_object(Bucket=bucket, Key=key)
            return int(resp["ContentLength"])
        except ClientError as e:
            if e.response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404:
                return None
            raise
    except RuntimeError:
        return None


def ls_prefix_exact(s3_uri: str) -> list[LsEntry]:
    """``ls`` of an exact object URI (no trailing slash appended)."""
    result, last_err, not_found = run_cli_candidates(
        (
            ("s5cmd", ["s5cmd", "ls", s3_uri]),
            ("aws", ["aws", "s3", "ls", s3_uri]),
        )
    )
    if result is not None:
        return _parse_cli_ls(str(result.stdout), result.tool)
    if not_found is not None:
        return []
    if last_err is not None:
        raise RuntimeError(f"[arcstore] ls {s3_uri} failed: {last_err}")
    return _boto3_ls_prefix(s3_uri)


def read_object_bytes(s3_uri: str) -> bytes:
    """Read an object straight into memory: s5cmd cat -> aws cp - -> boto3.

    The CLI-first order matters: training images often ship ``s5cmd`` but
    not ``boto3``, so a boto3-only read would spuriously fail there. Raises
    ``FileNotFoundError`` for a missing object.
    """
    result, _last_err, not_found = run_cli_candidates(
        (
            ("s5cmd", ["s5cmd", "cat", s3_uri]),
            ("aws", ["aws", "s3", "cp", s3_uri, "-"]),
        ),
        text=False,
    )
    if result is not None:
        return bytes(result.stdout)
    if not_found is not None:
        raise FileNotFoundError(f"{s3_uri} ({not_found})")

    import boto3
    from botocore.exceptions import ClientError

    bucket, key = split_s3(s3_uri)
    try:
        return boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
    except ClientError as e:
        if e.response.get("Error", {}).get("Code", "") in ("404", "NoSuchKey"):
            raise FileNotFoundError(s3_uri) from e
        raise


def download_file(s3_uri: str, local_path: str, *, label: str = "arcstore") -> None:
    """Download one object: s5cmd -> aws CLI -> boto3.

    Raises ``FileNotFoundError`` when the object is missing (so callers can
    treat S3 like a local filesystem) and ``RuntimeError`` on other failures.
    """
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)

    candidates: list[tuple[str, list[str]]] = []
    if have_s5cmd():
        candidates.append(("s5cmd", ["s5cmd", "cp", s3_uri, local_path]))
    if have_aws():
        candidates.append(("aws", ["aws", "s3", "cp", s3_uri, local_path]))

    result, last_err, not_found = run_cli_candidates(candidates)
    if result is not None:
        return
    if not_found is not None:
        raise FileNotFoundError(
            f"{label}: S3 object does not exist: {s3_uri!r} ({not_found})"
        )

    try:
        import boto3
        from botocore.exceptions import ClientError

        bucket, key = split_s3(s3_uri)
        try:
            boto3.client("s3").download_file(bucket, key, local_path)
            return
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchKey"):
                raise FileNotFoundError(
                    f"{label}: S3 object does not exist: {s3_uri!r}"
                ) from e
            raise
    except FileNotFoundError:
        raise
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"{label}: failed to download {s3_uri!r} -> {local_path!r}: "
            f"{last_err or e}"
        ) from e
