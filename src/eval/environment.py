"""Environment probes for reproducibility tracking.

Everything here is best-effort: if a probe fails (e.g. git is not on
PATH), the helper returns None/empty string and the caller records the
missing field without failing the run.
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
from pathlib import Path
from typing import Optional


def _run(cmd: list[str], timeout: float = 5.0) -> Optional[str]:
    try:
        out = subprocess.check_output(
            cmd, stderr=subprocess.DEVNULL, timeout=timeout,
        )
        return out.decode("utf-8", errors="replace").strip()
    except Exception:
        return None


def git_commit_sha(short: bool = True) -> Optional[str]:
    """Best-effort: returns short SHA of HEAD, or None if not a git checkout."""
    args = ["git", "rev-parse", "--short" if short else "--verify", "HEAD"]
    sha = _run(args)
    return sha or None


def git_is_dirty() -> Optional[bool]:
    """True if the working tree has uncommitted changes."""
    out = _run(["git", "status", "--porcelain"])
    if out is None:
        return None
    return bool(out)


def ffmpeg_version() -> Optional[str]:
    """Return the first line of ``ffmpeg -version``, or None if missing."""
    out = _run(["ffmpeg", "-version"])
    if not out:
        return None
    # First line looks like: "ffmpeg version 6.1 Copyright ..."
    first = out.splitlines()[0]
    m = re.search(r"ffmpeg version (\S+)", first)
    return m.group(1) if m else first


def file_sha256(path: str, chunk: int = 1 << 20) -> Optional[str]:
    """Return the SHA-256 hex digest of a file, or None if it does not exist."""
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    try:
        with p.open("rb") as f:
            while True:
                buf = f.read(chunk)
                if not buf:
                    break
                h.update(buf)
    except Exception:
        return None
    return h.hexdigest()


def collect_environment(input_video_path: Optional[str] = None) -> dict:
    """One-call snapshot for ``session_config.json``."""
    info = {
        "git_commit_sha": git_commit_sha(),
        "git_is_dirty": git_is_dirty(),
        "ffmpeg_version": ffmpeg_version(),
        "input_video_path": input_video_path or "",
        "input_video_sha256": (
            file_sha256(input_video_path) if input_video_path else None
        ),
        "cwd": os.getcwd(),
    }
    return info
