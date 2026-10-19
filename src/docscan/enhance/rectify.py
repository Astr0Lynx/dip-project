"""
Member B owns this package.  PUBLIC CONTRACT:

    rectify(scene_bgr, det, cfg) -> RectifiedDoc

Warps the quad of `det` to an upright rectangle. Output is BGR uint8, long side
<= cfg.rectify.max_side. Because the detector cannot know which side is "up", the
result may be rotated by a multiple of 90 degrees relative to the printed page.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..geometry import order_quad, quad_target_size
from ..types import Detection, RectifiedDoc
from ..validate import validate_rectified


def rectify(scene_bgr: np.ndarray, det: Detection, cfg) -> RectifiedDoc:
    # BASIC WORKING VERSION (B hardens: degenerate corners, border handling, tests)
    quad = order_quad(det.quad)
    w, h = quad_target_size(quad, cfg.rectify.max_side, cfg.rectify.min_side)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    H = cv2.getPerspectiveTransform(quad, dst)
    img = cv2.warpPerspective(scene_bgr, H, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    out = RectifiedDoc(det_id=det.det_id, image=img, H=H.astype(np.float64), quad=quad)
    if cfg.debug_validate:
        validate_rectified(out)
    return out
