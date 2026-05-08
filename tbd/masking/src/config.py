from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import Any


_FLOAT_RESOLVER_PATTERN = re.compile(
    r"""^(?:
    [-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |\.[0-9][0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?[0-9][0-9_]*(?:[eE][-+]?[0-9]+)
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+[.][0-9_]*
    |[-+]?[.](?:inf|Inf|INF)
    |[.](?:nan|NaN|NAN)
    )$""",
    re.X,
)


def _load_yaml_module():
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required. Install `pyyaml` before running this pipeline.") from exc
    return yaml


def _build_config_loader(yaml_module):
    class ConfigLoader(yaml_module.SafeLoader):
        pass

    ConfigLoader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        _FLOAT_RESOLVER_PATTERN,
        list("-+0123456789."),
    )
    return ConfigLoader


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
    loader = _build_config_loader(yaml)
    path = Path(path)
    raw = yaml.load(path.read_text(encoding="utf-8"), Loader=loader) or {}
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
