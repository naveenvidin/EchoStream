"""config_writer.py — writes a temporary runtime config from UI form values.

Loads default.json as the base, applies only the fields the user touched,
and writes the result to a temp file. default.json is never mutated.

Usage:
    from src.ui.config_writer import write_runtime_config

    path = write_runtime_config(
        default_config_path="configs/default.json",
        input_source="0",           # "0" for live camera, or a file path
        loop_video=False,
        classes="person",
        baseline_enabled=True,
        save_artifacts=True,
        output_dir="runs/my_session",
    )
    # path is a pathlib.Path to the temp config file
"""
from __future__ import annotations

import copy
import json
import tempfile
import time
from pathlib import Path


def write_runtime_config(
    default_config_path: str | Path,
    input_source: str,
    loop_video: bool,
    classes: str,
    baseline_enabled: bool,
    save_artifacts: bool,
    output_dir: str | None,
    model: str | None = None,
) -> Path:
    """Merge UI values into a copy of default.json and write to a temp file.

    Returns the Path to the temp config file. The caller is responsible for
    deleting it after the session ends (or it will be cleaned up by the OS
    on the next reboot via /tmp).
    """
    default_config_path = Path(default_config_path)
    with open(default_config_path, "r") as f:
        config = json.load(f)

    # Deep copy so we never mutate the loaded dict (defensive, not strictly
    # necessary since we loaded fresh, but makes intent clear).
    config = copy.deepcopy(config)

    # Resolve output dir — default to runs/<timestamp> if blank or None.
    if save_artifacts and not output_dir:
        output_dir = str(Path("runs") / time.strftime("session_%Y%m%d_%H%M%S"))
    model = (model or "").strip() or None

    # --- camera_h264 overrides ---
    cam = config.setdefault("camera_h264", {})
    cam["input"] = input_source
    cam["loop_video"] = loop_video
    cam["classes"] = classes
    if model is not None:
        cam["model"] = model
    cam["baseline_enabled"] = baseline_enabled
    cam["save_artifacts"] = save_artifacts
    cam["output_dir"] = output_dir if save_artifacts else None

    # --- server_h264 overrides (both adaptive and baseline share same config) ---
    srv = config.setdefault("server_h264", {})
    srv["classes"] = classes
    if model is not None:
        srv["model"] = model
    srv["save_artifacts"] = save_artifacts
    srv["output_dir"] = output_dir if save_artifacts else None
    # Never open cv2 windows when launched from the UI.
    srv["show_window"] = False

    # Write to a named temp file that persists until explicitly deleted.
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix="echostream_runtime_",
        delete=False,
    )
    json.dump(config, tmp, indent=2)
    tmp.close()

    return Path(tmp.name)