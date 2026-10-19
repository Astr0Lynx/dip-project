"""
Member A owns this package.  PUBLIC CONTRACT (signatures frozen, bodies are STUBS):

    find_candidates(scene_bgr, cfg)              -> List[Candidate]
    classify_candidates(scene_bgr, cands, cfg)   -> List[Detection]   # ALL candidates, with accepted flag
    detect_all(scene_bgr, cfg)                   -> List[Detection]   # find + classify
    detect_documents(scene_bgr, cfg)             -> List[Detection]   # only accepted ones

Rules every implementation must obey:
  * input is the FULL-RES scene; internal work may use a downscaled copy but every
    returned coordinate must be mapped back to full-res.
  * det_id values are unique within the scene (0..n-1).
  * quads are ordered with geometry.order_quad and are float32 (4,2).
  * the list returned by detect_all contains rejected candidates too (accepted=False);
    the evaluation code needs them to compute rejection precision/recall.
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..types import Candidate, Detection
from ..validate import validate_detections
from . import candidates as _candidates


def find_candidates(scene_bgr: np.ndarray, cfg) -> List[Candidate]:
    """Candidate document quads (PLAN A2): contours of three edge maps + Hough fallback."""
    return _candidates.find_candidates(scene_bgr, cfg)


def classify_candidates(scene_bgr: np.ndarray, cands: List[Candidate], cfg) -> List[Detection]:
    # STUB until A4 (cascade): accepts everything.
    return [Detection(det_id=i, quad=c.quad, score=1.0, accepted=True, decided_by="heuristic",
                      source=c.source, contour=c.contour, quad_fit=c.quad_fit)
            for i, c in enumerate(cands)]


def detect_all(scene_bgr: np.ndarray, cfg) -> List[Detection]:
    dets = classify_candidates(scene_bgr, find_candidates(scene_bgr, cfg), cfg)
    if cfg.debug_validate:
        validate_detections(dets, scene_bgr.shape[:2])
    return dets


def detect_documents(scene_bgr: np.ndarray, cfg) -> List[Detection]:
    return [d for d in detect_all(scene_bgr, cfg) if d.accepted]
