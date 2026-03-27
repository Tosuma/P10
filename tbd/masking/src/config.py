from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


def _load_yaml_module():
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required. Install `pyyaml` before running this pipeline.") from exc
    return yaml


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    yaml = _load_yaml_module()
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "base_config" in raw:
        base_path = (path.parent / raw["base_config"]).resolve()
        base_cfg = load_config(base_path)
        raw = deep_update(base_cfg, {k: v for k, v in raw.items() if k != "base_config"})
    raw["_config_path"] = str(path.resolve())
    return raw


def dump_config(path: str | Path, config: dict[str, Any]) -> None:
    yaml = _load_yaml_module()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
