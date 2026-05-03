from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def load_json_config(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def apply_config_defaults(parser, config: Dict[str, Any], section: str) -> None:
    """Apply config values as argparse defaults.

    CLI flags still win because argparse uses explicit args over defaults.
    """
    if not config:
        return
    block = config.get(section)
    if not isinstance(block, dict):
        return
    # Only apply keys that correspond to argparse dest names.
    safe: Dict[str, Any] = {}
    for action in getattr(parser, "_actions", []):
        dest = getattr(action, "dest", None)
        if not dest or dest == "help":
            continue
        if dest in block:
            safe[dest] = block[dest]
    if safe:
        parser.set_defaults(**safe)

