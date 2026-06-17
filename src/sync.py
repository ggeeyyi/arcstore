"""arcstore-sync — bidirectional local<->S3 *code* sync for the koala workflow.

Ported from ``arc_toolkit.sync``. The cluster pulls your code from S3 at submit
time (``koala submit --code``). To deploy the *latest* code reliably you need a
MIRROR (so files deleted locally are also removed on S3 — otherwise the pod
keeps stale modules and breaks subtly):

    s3://<bucket>/<user>/code/<project>/      MIRROR of local code  (push --delete)

arcstore-sync syncs **code only**. Run outputs (checkpoints / logs / artifacts)
are NOT its job — write them to a *different* S3 location (logs via LogTee,
checkpoints via CheckpointManager's ``s3_prefix``, anything else via plain
s5cmd / arcstore.put), never under ``code/<project>/``, so the code mirror's
``--delete`` can never touch them. Data/output/cache dirs and ``.git`` are
excluded from the push (S3 hates many tiny files — a ``.git`` tree is thousands
of them).

  ``push``   local -> code/  (mirror; previews + confirms deletions)
  ``pull``   code/ -> local  (non-destructive; never deletes local files)
  ``status`` show the resolved code prefix + local git state (no S3 calls)

Config via env: ``ARCSTORE_SYNC_BUCKET``, ``KOALA_USER``/``USER``,
``ARCSTORE_CODE_S3``, ``ARCSTORE_SYNC_PROJECT``, ``ARCSTORE_SYNC_EXCLUDE``
(comma-separated, appended to the defaults). Requires ``s5cmd`` on PATH.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_BUCKET = "s3://arcwm-code-us-west-2"

# Never mirror these. arcstore-sync runs s5cmd from the repo root with a relative
# source, so patterns match the *relative* path and `*` spans `/`. For each
# cache/output dir we list BOTH `dir/*` (top-level) and `*/dir/*` (nested — e.g.
# inside a vendored submodule). .git is excluded hard (a tree of thousands of
# tiny objects S3 throttles on); run outputs/data belong on a SEPARATE prefix,
# never under the code mirror.
DEFAULT_EXCLUDES = [
    # VCS metadata (dir + nested dir + submodule gitlink files)
    ".git",
    ".git/*",
    "*/.git",
    "*/.git/*",
    # virtualenvs / tooling caches / build artifacts (top-level + nested)
    ".venv/*",
    "*/.venv/*",
    "__pycache__/*",
    "*/__pycache__/*",
    "*.pyc",
    ".ruff_cache/*",
    "*/.ruff_cache/*",
    ".pytest_cache/*",
    "*/.pytest_cache/*",
    ".mypy_cache/*",
    "*/.mypy_cache/*",
    "*.egg-info/*",
    "*/*.egg-info/*",
    "dist/*",
    "*/dist/*",
    "build/*",
    "*/build/*",
    "node_modules/*",
    "*/node_modules/*",
    # editor / OS cruft
    ".DS_Store",
    "*/.DS_Store",
    ".claude/*",
    ".cursor/*",
    # run outputs & data — these go to a SEPARATE S3 prefix, never the code mirror
    "wandb/*",
    "*/wandb/*",
    "outputs/*",
    "*/outputs/*",
    "exp/*",
    "exp_*/*",
    "*/exp/*",
    # secrets
    ".env",
    "*/.env",
]

__all__ = ["main"]


@dataclass
class Config:
    """Resolved arcstore-sync settings: which local tree mirrors to which S3 code prefix."""

    project: str
    user: str
    code: str
    excludes: list[str]
    root: Path


def _project_name(root: Path) -> str:
    pp = root / "pyproject.toml"
    if pp.exists():
        m = re.search(r'(?m)^\s*name\s*=\s*"([^"]+)"', pp.read_text())
        if m:
            return m.group(1)
    return root.name


def resolve_config(root: Path, env: dict) -> Config:
    """Build a Config from env vars (ARCSTORE_SYNC_*/KOALA_USER/USER) + the project name."""
    project = env.get("ARCSTORE_SYNC_PROJECT") or _project_name(root)
    user = env.get("KOALA_USER") or env.get("USER") or "unknown"
    bucket = env.get("ARCSTORE_SYNC_BUCKET", DEFAULT_BUCKET).rstrip("/")
    code = (env.get("ARCSTORE_CODE_S3") or f"{bucket}/{user}/code/{project}").rstrip("/")
    excludes = list(DEFAULT_EXCLUDES)
    excludes += [e.strip() for e in env.get("ARCSTORE_SYNC_EXCLUDE", "").split(",") if e.strip()]
    return Config(project, user, code, excludes, root)


def empty_submodules(root: Path) -> list[str]:
    """Submodule paths (from .gitmodules) whose working tree is missing/empty.

    Pushing those to S3 would ship the cluster an empty dir and break ``uv sync``.
    """
    gm = root / ".gitmodules"
    if not gm.exists():
        return []
    out = []
    for path in re.findall(r"(?m)^\s*path\s*=\s*(.+)$", gm.read_text()):
        d = root / path.strip()
        if not d.is_dir() or not any(d.iterdir()):
            out.append(path.strip())
    return out


def _exclude_args(excludes: list[str]) -> list[str]:
    args: list[str] = []
    for e in excludes:
        args += ["--exclude", e]
    return args


def _s5cmd() -> str:
    exe = shutil.which("s5cmd")
    if not exe:
        sys.exit("[arcstore-sync] s5cmd not found on PATH — install it (brew install peak/tap/s5cmd).")
    return exe


def _confirm(prompt: str) -> bool:
    try:
        return input(prompt).strip().lower() in ("y", "yes")
    except EOFError:
        return False


def cmd_push(cfg: Config, args) -> None:
    """push: mirror local code -> S3 code/ prefix (previews + confirms any deletions)."""
    # Guard: an un-checked-out submodule would push an empty vendor dir to S3,
    # which then breaks `uv sync` on the pod. Refuse with the fix.
    empty = empty_submodules(cfg.root)
    if empty and not args.allow_empty_submodules:
        sys.exit(
            "[arcstore-sync] REFUSING push: submodule(s) not checked out: "
            + ", ".join(empty)
            + "\n    run: git submodule update --init   (then push again)"
        )

    s5 = _s5cmd()
    # Run s5cmd FROM cfg.root with a relative "." source. s5cmd matches --exclude
    # patterns against the source path, so an absolute source (".../repo/") would
    # never match patterns like "exp/*"; a relative root makes the excludes bite.
    src = "."
    dst = cfg.code + "/"
    excl = _exclude_args(cfg.excludes)
    print(f"[arcstore-sync] MIRROR {cfg.root}/ -> {dst}")

    # Preview the deletions a mirror would make (best-effort parse of --dry-run),
    # so a mispointed prefix that would wipe non-code files is caught before it runs.
    dry = subprocess.run(
        [s5, "--dry-run", "sync", "--delete", *excl, src, dst],
        cwd=str(cfg.root),
        capture_output=True,
        text=True,
    )
    deletes = [ln for ln in dry.stdout.splitlines() if ln.lower().startswith("rm ")]
    if deletes:
        print(f"[arcstore-sync] {len(deletes)} stale remote file(s) will be DELETED:")
        for ln in deletes[:20]:
            print("    " + ln)
        if len(deletes) > 20:
            print(f"    ... and {len(deletes) - 20} more")
        if not args.yes and not _confirm("[arcstore-sync] proceed with mirror (incl. deletes)? [y/N] "):
            sys.exit("[arcstore-sync] aborted.")

    cmd = [s5, "sync", "--delete", *excl, src, dst]
    if args.dry_run:
        print("[arcstore-sync] (dry-run) would run:", " ".join(cmd), f"(cwd={cfg.root})")
        return
    subprocess.run(cmd, cwd=str(cfg.root), check=True)
    print(
        f'[arcstore-sync] done. submit with:  koala submit ... --code "{cfg.code}:/data/work/run_codes"'
    )


def cmd_pull(cfg: Config, args) -> None:
    """pull: S3 code/ -> local, non-destructive (never deletes local files)."""
    s5 = _s5cmd()
    dst = str(cfg.root).rstrip("/") + "/"
    # NON-destructive: no --delete, so local-only files (uncommitted work) survive.
    cmd = [s5, "sync", cfg.code + "/*", dst]
    if args.dry_run:
        cmd = [s5, "--dry-run", *cmd[1:]]
    print(f"[arcstore-sync] {cfg.code}/ -> {dst} (non-destructive; never deletes local files)")
    subprocess.run(cmd, check=True)


def cmd_status(cfg: Config, args) -> None:
    """status: print the resolved code prefix + local git state (no S3 calls)."""
    print(f"project:  {cfg.project}")
    print(f"code:     {cfg.code}   (run outputs go elsewhere — never under this prefix)")
    empty = empty_submodules(cfg.root)
    if empty:
        print(f"submodules not synced: {', '.join(empty)}  (git submodule update --init)")
    print("-- local git --")
    subprocess.run(["git", "-C", str(cfg.root), "status", "-sb"])


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="arcstore-sync",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("push", help="mirror local code -> S3 (deletions propagate)")
    pp.add_argument("--dry-run", action="store_true", help="show what would happen, transfer nothing")
    pp.add_argument("-y", "--yes", action="store_true", help="skip the delete confirmation")
    pp.add_argument(
        "--allow-empty-submodules", action="store_true", help="push even if a submodule is empty"
    )

    pl = sub.add_parser("pull", help="S3 code -> local (non-destructive)")
    pl.add_argument("--dry-run", action="store_true")

    sub.add_parser("status", help="show resolved code prefix + local git state")

    args = p.parse_args(argv)
    cfg = resolve_config(Path.cwd(), dict(os.environ))
    {"push": cmd_push, "pull": cmd_pull, "status": cmd_status}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
