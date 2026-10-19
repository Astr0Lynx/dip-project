"""
docscan.validate  --  FROZEN contract checkers (v1.0).

Each stage should call the validator for the object it RETURNS when
cfg.debug_validate is True. This catches interface bugs at the source instead of
three modules later.
"""
from __future__ import annotations

import numpy as np

from .types import (BINARIZE_METHODS, DECIDERS, ILLUM_METHODS, SOURCES,
                    BinarizedDoc, Detection, EnhancedDoc, RectifiedDoc)


class ContractError(ValueError):
    """Raised when an object violates the shared data contract."""


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise ContractError(msg)


def validate_quad(q, name: str = "quad") -> None:
    _need(isinstance(q, np.ndarray), f"{name}: must be np.ndarray")
    _need(q.shape == (4, 2), f"{name}: shape must be (4,2), got {q.shape}")
    _need(q.dtype == np.float32, f"{name}: dtype must be float32, got {q.dtype}")
    _need(bool(np.all(np.isfinite(q))), f"{name}: contains NaN/inf")


def validate_detection(d: Detection) -> None:
    # Python scalars only: numpy scalars (np.int64 from argsort, np.bool_ from
    # `p >= thresh`) are rejected on purpose -- cast with int() / bool() / float().
    _need(isinstance(d.det_id, int) and d.det_id >= 0,
          f"det_id must be a Python int >= 0, got {d.det_id!r} ({type(d.det_id).__name__}); cast with int()")
    validate_quad(d.quad, "Detection.quad")
    _need(0.0 <= d.score <= 1.0, f"score out of [0,1]: {d.score}")
    _need(isinstance(d.accepted, bool),
          f"accepted must be a Python bool, got {type(d.accepted).__name__}; cast with bool()")
    _need(d.decided_by in DECIDERS, f"decided_by must be in {DECIDERS}")
    _need(d.source in SOURCES, f"source must be in {SOURCES}")
    _need(0.0 <= d.quad_fit <= 1.0, "quad_fit out of [0,1]")
    _need(0.0 <= d.occlusion_est <= 1.0, "occlusion_est out of [0,1]")
    if d.contour is not None:
        _need(d.contour.ndim == 3 and d.contour.shape[1:] == (1, 2),
              "contour must have shape (N,1,2)")
    if d.features is not None:
        _need(d.features.ndim == 1 and d.features.dtype == np.float32,
              "features must be 1-D float32")


def validate_detections(ds, scene_hw=None) -> None:
    ids = [d.det_id for d in ds]
    _need(len(ids) == len(set(ids)), "det_id values must be unique within a scene")
    for d in ds:
        validate_detection(d)
        if scene_hw is not None:
            h, w = scene_hw
            # allow a small margin: quads may poke slightly outside the frame
            _need(bool(np.all(d.quad[:, 0] > -0.1 * w) and np.all(d.quad[:, 0] < 1.1 * w)
                       and np.all(d.quad[:, 1] > -0.1 * h) and np.all(d.quad[:, 1] < 1.1 * h)),
                  f"det {d.det_id}: quad far outside the scene (wrong coordinate frame?)")


def _validate_img(a: np.ndarray, channels: int | None, name: str) -> None:
    _need(isinstance(a, np.ndarray) and a.dtype == np.uint8, f"{name}: must be uint8 ndarray")
    if channels is None:
        _need(a.ndim == 2, f"{name}: must be 2-D gray, got shape {a.shape}")
    else:
        _need(a.ndim == 3 and a.shape[2] == channels, f"{name}: must be (H,W,{channels}), got {a.shape}")


def validate_rectified(r: RectifiedDoc) -> None:
    _validate_img(r.image, 3, "RectifiedDoc.image")
    _need(r.H.shape == (3, 3) and r.H.dtype == np.float64, "H must be (3,3) float64")
    validate_quad(r.quad, "RectifiedDoc.quad")


def validate_enhanced(e: EnhancedDoc) -> None:
    _validate_img(e.gray, None, "EnhancedDoc.gray")
    _need(e.illum_method in ILLUM_METHODS, f"illum_method must be in {ILLUM_METHODS}")
    if e.shadow_map is not None:
        _validate_img(e.shadow_map, None, "EnhancedDoc.shadow_map")
        _need(e.shadow_map.shape == e.gray.shape, "shadow_map must match gray shape")


def validate_binarized(b: BinarizedDoc) -> None:
    _validate_img(b.binary, None, "BinarizedDoc.binary")
    vals = np.unique(b.binary)
    _need(bool(np.all(np.isin(vals, (0, 255)))), f"binary must contain only 0/255, got {vals[:5]}")
    _need(b.method in BINARIZE_METHODS, f"method must be in {BINARIZE_METHODS}")
