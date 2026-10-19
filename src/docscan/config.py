"""
docscan.config  --  FROZEN loader (v1.0). The VALUES live in configs/default.yaml.

Ownership of YAML sections (edit only your own section to avoid merge conflicts):
  Member A : detect, cascade, synth
  Member B : overlap, rectify, illum, binarize, eval
  Shared   : seed, debug_validate, paths   (change only via Contract Change Request)

Usage:
    cfg = load_config()                              # defaults
    cfg = load_config(overrides={"detect.max_side": 800})
    cfg.detect.max_side  -> 800
    resolve_path(cfg.cascade.ml.model_path)          # -> <repo>/models/cascade_ml.joblib

(v1.1) Relative paths in the YAML (paths.*, *.model_path, ...) are relative to the
REPO ROOT, not to the current working directory. Always open them through
resolve_path(); IDEs (e.g. PyCharm) run scripts with cwd = the script's folder.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]   # requires `pip install -e .` (editable)
_DEFAULT = REPO_ROOT / "configs" / "default.yaml"


def resolve_path(p: Union[str, Path]) -> Path:
    """Absolute paths are returned unchanged; relative ones are anchored at REPO_ROOT."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


class Config:
    """Recursive attribute-access wrapper around a dict."""

    def __init__(self, d: Dict[str, Any]):
        for k, v in d.items():
            setattr(self, k, Config(v) if isinstance(v, dict) else v)

    def to_dict(self) -> Dict[str, Any]:
        return {k: (v.to_dict() if isinstance(v, Config) else v) for k, v in self.__dict__.items()}

    def get(self, key: str, default: Any = None) -> Any:
        return self.__dict__.get(key, default)

    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"


def _deep_set(d: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def load_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Config:
    with open(path or _DEFAULT, "r") as f:
        raw = yaml.safe_load(f)
    for dotted, value in (overrides or {}).items():
        _deep_set(raw, dotted, value)
    return Config(raw)
