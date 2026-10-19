"""docscan.viz -- debug overlays. Owner: B. Signature frozen; A may import it freely."""
from __future__ import annotations

from typing import List

import cv2
import numpy as np

from .types import Detection


def draw_detections(scene_bgr: np.ndarray, dets: List[Detection]) -> np.ndarray:
    """Return a copy of the scene with quads drawn: green = accepted, red = rejected,
    labelled `id:score:stage`."""
    out = scene_bgr.copy()
    for d in dets:
        color = (0, 200, 0) if d.accepted else (0, 0, 220)
        pts = np.round(d.quad).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], True, color, 3, cv2.LINE_AA)
        x, y = int(d.quad[0, 0]), int(d.quad[0, 1])
        cv2.putText(out, f"{d.det_id}:{d.score:.2f}:{d.decided_by[0]}", (x + 4, max(14, y + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out
