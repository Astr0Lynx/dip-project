"""
Member B owns this package (STRETCH goal).  PUBLIC CONTRACT (v1.1):

    resolve_overlaps(scene_bgr, detections, cfg, classify=None) -> List[Detection]

Input : ALL detections of the scene (accepted AND rejected), exactly as detect_all returns them.
Output: ALL detections after repair, where
          * ACCEPTED detections may get a repaired quad (hidden corners completed by line
            extension), an updated quad_fit, occlusion_est and source="overlap_resolved";
            their det_id does not change;
          * a merged blob (two touching documents in one contour), accepted OR rejected, may be
            REPLACED by split children: the parent is removed, every child gets a new det_id
            starting at max(all input det_ids) + 1 and source="overlap_resolved";
          * children are judged by `classify` (A's detect.api.classify_candidates, injected by
            the pipeline so this package never imports detect/): classify(scene, cands, cfg)
            -> List[Detection], det_ids local to that call (re-number them). If classify is
            None (oracle runs), children of an ACCEPTED parent inherit accepted=True,
            score and decided_by; children of a rejected parent are not created;
          * rejected detections that are not split are returned unchanged.
Must be the identity when cfg.overlap.enabled is False, must never raise on valid input
(log and return the input unchanged on failure), and det_ids must stay unique.
Why ALL detections: the cascade usually REJECTS merged blobs (for two similar-sized documents
the blob's IoU with either one is at most about 0.5, so blobs are negatives in training), so an
accepted-only stage would never see the blobs it is supposed to split.
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional

import numpy as np

from ..types import Candidate, Detection

Classifier = Callable[[np.ndarray, List[Candidate], Any], List[Detection]]


def resolve_overlaps(scene_bgr: np.ndarray, detections: List[Detection], cfg,
                     classify: Optional[Classifier] = None) -> List[Detection]:
    # STUB (B replaces): identity.
    return list(detections)
