"""
Member B owns this file. STUB.  Loader + degradation tools for DIBCO.

PUBLIC CONTRACT (used only by B):
    list_dibco(root, years=None)  -> List[DibcoItem]     # (name, year, img_path, gt_path)
    load_dibco(item)              -> (gray uint8, gt uint8 {0,255} with 0=ink)
    degrade_illumination(gray, rng, severity) -> gray    # synthetic shadows / gradients
See PLAN section 7 (B3, B6).
"""
from __future__ import annotations


def list_dibco(root, years=None):
    raise NotImplementedError("Member B: implement (PLAN B3)")


def load_dibco(item):
    raise NotImplementedError("Member B: implement (PLAN B3)")


def degrade_illumination(gray, rng, severity=1.0):
    raise NotImplementedError("Member B: implement (PLAN B3)")
