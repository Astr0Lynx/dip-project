"""Shared (frozen) helper behaviour added in contract v1.1."""
from pathlib import Path

import numpy as np
import pytest
from docscan.config import REPO_ROOT, resolve_path
from docscan.io_utils import read_json, write_json
from docscan.types import Detection
from docscan.validate import ContractError, validate_detection


def test_write_json_accepts_numpy_scalars(tmp_path):
    p = tmp_path / "x.json"
    write_json(p, {"b": np.float64(0.7) >= 0.5, "i": np.int64(3), "f": np.float32(0.25)})
    assert read_json(p) == {"b": True, "i": 3, "f": 0.25}


def test_resolve_path_anchors_relative_paths_at_repo_root():
    assert resolve_path("models/cascade_ml.joblib") == REPO_ROOT / "models" / "cascade_ml.joblib"
    assert (REPO_ROOT / "configs" / "default.yaml").exists()
    absolute = Path("/tmp/somewhere/file.txt")
    assert resolve_path(absolute) == absolute


def test_validator_explains_numpy_scalars():
    q = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], np.float32)
    d = Detection(det_id=0, quad=q, score=0.9, accepted=np.float64(0.7) >= 0.5,
                  decided_by="ml", source="contour")
    with pytest.raises(ContractError, match="cast with bool"):
        validate_detection(d)
    d.accepted, d.det_id = True, np.int64(0)
    with pytest.raises(ContractError, match="cast with int"):
        validate_detection(d)
