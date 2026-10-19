"""
Member B owns this file.  PUBLIC CONTRACT:

    correct_illumination(rect, cfg) -> EnhancedDoc

Converts the rectified colour crop to gray and removes uneven lighting / shadows.
Method chosen by cfg.illum.method in {"none","bgsub","homomorphic"}.
"""
from __future__ import annotations

import cv2

from ..types import EnhancedDoc, RectifiedDoc
from ..validate import validate_enhanced


def correct_illumination(rect: RectifiedDoc, cfg) -> EnhancedDoc:
    # STUB (B replaces): gray conversion only.
    gray = cv2.cvtColor(rect.image, cv2.COLOR_BGR2GRAY)
    out = EnhancedDoc(det_id=rect.det_id, gray=gray, illum_method="none", shadow_map=None)
    if cfg.debug_validate:
        validate_enhanced(out)
    return out
