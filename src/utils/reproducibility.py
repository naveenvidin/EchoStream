"""Deterministic-run helpers with read-back validation.

`set_seeds(seed, strict)` seeds Python / NumPy / Torch, disables CuDNN
autotuning, and — crucially — reads back the actual state so the caller
can tell whether the strict flags took effect. This matters because:

- On CPU-only installs, CuDNN flags are ignored.
- On CUDA, `torch.use_deterministic_algorithms(True)` raises at runtime
  if an op has no deterministic impl; we capture that.
- `CUBLAS_WORKSPACE_CONFIG` must be set *before* CUDA init.

The returned dict is persisted into `session_config.json` alongside each
benchmark run so results can be audited later.
"""
from __future__ import annotations

import os
import platform
import random
from typing import Any, Dict


def _capture_torch_state() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        info["cudnn_version"] = (
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available() else None
        )
        info["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
        info["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
        info["mps_available"] = bool(
            getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
        )
        try:
            info["deterministic_algorithms"] = bool(
                torch.are_deterministic_algorithms_enabled()
            )
        except AttributeError:
            info["deterministic_algorithms"] = False
    except Exception as e:
        info["torch_error"] = repr(e)
    return info


def set_seeds(seed: int = 0, strict: bool = False) -> Dict[str, Any]:
    """Seed RNGs and return a structured report of what actually stuck.

    Returns a dict with:
      requested: the raw args
      effective: what the runtime reports now
      warnings:  things the caller should know (e.g. strict requested on CPU)
      strict_achieved: bool — True iff strict was both requested and enforced
    """
    warnings: list[str] = []
    requested = {"seed": int(seed), "strict": bool(strict)}

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
        numpy_ok = True
    except Exception as e:
        numpy_ok = False
        warnings.append(f"numpy seed failed: {e!r}")

    torch_ok = False
    strict_enforced = False
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if strict:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            try:
                torch.use_deterministic_algorithms(True)
                strict_enforced = True
            except Exception as e:
                warnings.append(
                    f"strict determinism requested but could not be enforced: {e!r}"
                )
        torch_ok = True
    except Exception as e:
        warnings.append(f"torch seed failed: {e!r}")

    effective = _capture_torch_state()
    effective["python_hash_seed"] = os.environ.get("PYTHONHASHSEED")
    effective["python_version"] = platform.python_version()
    effective["platform"] = platform.platform()

    if strict and not strict_enforced:
        warnings.append("strict=True requested; falling back to non-strict")

    return {
        "requested": requested,
        "effective": effective,
        "warnings": warnings,
        "numpy_ok": numpy_ok,
        "torch_ok": torch_ok,
        "strict_achieved": bool(strict and strict_enforced),
    }
