"""docscan.io_utils -- FROZEN small I/O helpers (v1.0)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def read_image(path) -> np.ndarray:
    """Read as BGR uint8 (H,W,3). Raises FileNotFoundError if unreadable."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return img


def read_gray(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return img


def write_image(path, img: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img):
        raise IOError(f"cannot write image: {path}")


def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _json_default(o: Any):
    # np.bool_ is NOT a subclass of np.integer, so it needs its own branch
    # (e.g. `accepted = p >= thresh` with numpy floats yields np.bool_).
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_json(path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def read_json(path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def get_rng(seed: int) -> np.random.Generator:
    """All randomness in the project goes through a Generator created here."""
    return np.random.default_rng(seed)
