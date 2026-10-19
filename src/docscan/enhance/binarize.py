"""
Member B owns this file.  PUBLIC CONTRACT:

    binarize(enh, method, cfg) -> BinarizedDoc      # method in BINARIZE_METHODS
    binarize_all(enh, cfg)     -> Dict[str, BinarizedDoc]   # every method in cfg.binarize.methods

Output convention: uint8 {0,255}, 0 = ink, 255 = paper.
"""
from __future__ import annotations

from typing import Dict

import cv2

from ..types import BinarizedDoc, EnhancedDoc
from ..validate import validate_binarized


def binarize(enh: EnhancedDoc, method: str, cfg) -> BinarizedDoc:
    # STUB (B replaces): Otsu for every method name.
    _, b = cv2.threshold(enh.gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out = BinarizedDoc(det_id=enh.det_id, binary=b, method=method)
    if cfg.debug_validate:
        validate_binarized(out)
    return out


def binarize_all(enh: EnhancedDoc, cfg) -> Dict[str, BinarizedDoc]:
    return {m: binarize(enh, m, cfg) for m in cfg.binarize.methods}
