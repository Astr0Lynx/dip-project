"""
docscan.geometry  --  FROZEN shared helpers (contract v1.0).

Everything here is implemented and unit-tested already. Both members import
from this file instead of re-writing quad math. Bug fixes are allowed (tell the
other member); signature changes need a Contract Change Request.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .types import Quad


def order_quad(pts: Sequence) -> Quad:
    """Return 4 points as float32 (4,2), clockwise on screen, starting at the
    top-left-most point (smallest x+y; ties broken by smaller y)."""
    p = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    c = p.mean(axis=0)
    ang = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])  # ascending angle = clockwise (y down)
    p = p[np.argsort(ang)]
    key = p[:, 0] + p[:, 1]
    start = min(range(4), key=lambda k: (round(float(key[k]), 3), float(p[k, 1])))
    return np.roll(p, -start, axis=0).astype(np.float32)


def quad_area(q: Quad) -> float:
    """Shoelace area in px^2."""
    q = np.asarray(q, dtype=np.float64).reshape(4, 2)
    x, y = q[:, 0], q[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def is_convex_quad(q: Quad) -> bool:
    q = np.asarray(q, dtype=np.float64).reshape(4, 2)
    s = []
    for i in range(4):
        a = q[(i + 1) % 4] - q[i]
        b = q[(i + 2) % 4] - q[(i + 1) % 4]
        s.append(a[0] * b[1] - a[1] * b[0])
    s = np.asarray(s)
    return bool(np.all(s > 0) or np.all(s < 0))


def quad_edge_lengths(q: Quad) -> np.ndarray:
    """[top, right, bottom, left] = |TR-TL|, |BR-TR|, |BL-BR|, |TL-BL| (float64)."""
    q = np.asarray(q, dtype=np.float64).reshape(4, 2)
    return np.array([np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)])


def quad_edge_angles_deg(q: Quad) -> np.ndarray:
    """Direction of the 4 edges (TL->TR, TR->BR, BR->BL, BL->TL) in degrees, (-180,180]."""
    q = np.asarray(q, dtype=np.float64).reshape(4, 2)
    out = []
    for i in range(4):
        d = q[(i + 1) % 4] - q[i]
        out.append(np.degrees(np.arctan2(d[1], d[0])))
    return np.asarray(out)


def quad_target_size(q: Quad, max_side: int = 1600, min_side: int = 64) -> Tuple[int, int]:
    """Output (width, height) for rectifying quad q. Width = longer of top/bottom
    edges, height = longer of left/right edges, then scaled DOWN so the long side
    is <= max_side (memory bound) and each side is >= min_side."""
    e = quad_edge_lengths(q)
    w, h = max(e[0], e[2]), max(e[1], e[3])
    s = min(1.0, max_side / max(w, h, 1e-6))
    return max(int(round(w * s)), min_side), max(int(round(h * s)), min_side)


def scale_quad(q: Quad, sx: float, sy: float | None = None) -> Quad:
    sy = sx if sy is None else sy
    return (np.asarray(q, dtype=np.float32).reshape(4, 2) * np.array([sx, sy], np.float32)).astype(np.float32)


def quad_bbox(q: Quad) -> Tuple[int, int, int, int]:
    """Integer (x0, y0, x1, y1) bounding box, x1/y1 exclusive."""
    q = np.asarray(q).reshape(4, 2)
    return (int(np.floor(q[:, 0].min())), int(np.floor(q[:, 1].min())),
            int(np.ceil(q[:, 0].max())), int(np.ceil(q[:, 1].max())))


def quad_mask(shape_hw: Tuple[int, int], q: Quad) -> np.ndarray:
    """uint8 mask (H,W), 255 inside the quad."""
    m = np.zeros(shape_hw, np.uint8)
    cv2.fillConvexPoly(m, np.round(np.asarray(q, np.float32)).astype(np.int32).reshape(-1, 1, 2), 255)
    return m


def quad_iou(a: Quad, b: Quad) -> float:
    """IoU of two convex quads (0 if degenerate / disjoint)."""
    a = np.ascontiguousarray(a, dtype=np.float32).reshape(4, 2)
    b = np.ascontiguousarray(b, dtype=np.float32).reshape(4, 2)
    aa, ab = quad_area(a), quad_area(b)
    if aa < 1e-6 or ab < 1e-6:
        return 0.0
    inter, _ = cv2.intersectConvexConvex(a, b)
    inter = float(inter)
    union = aa + ab - inter
    return float(inter / union) if union > 1e-9 else 0.0


def cyclic_shifts(q: Quad) -> List[Quad]:
    q = np.asarray(q, dtype=np.float32).reshape(4, 2)
    return [np.roll(q, k, axis=0) for k in range(4)]


def best_cyclic_alignment(pred: Quad, gt: Quad) -> Tuple[Quad, float]:
    """Re-order `pred` (cyclic shift only) so its corners line up with `gt`.
    Returns (aligned_pred, mean_corner_distance_px)."""
    gt = np.asarray(gt, dtype=np.float32).reshape(4, 2)
    best, best_d = None, float("inf")
    for cand in cyclic_shifts(pred):
        d = float(np.linalg.norm(cand - gt, axis=1).mean())
        if d < best_d:
            best, best_d = cand, d
    return best, best_d


def match_quads(pred: Sequence[Quad], gt: Sequence[Quad],
                iou_thresh: float = 0.5) -> List[Tuple[int, int, float]]:
    """One-to-one matching (Hungarian on IoU). Returns [(pred_idx, gt_idx, iou)]
    for pairs with IoU >= iou_thresh. Order-independent; unmatched are omitted."""
    if len(pred) == 0 or len(gt) == 0:
        return []
    iou = np.zeros((len(pred), len(gt)), dtype=np.float64)
    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            iou[i, j] = quad_iou(p, g)
    r, c = linear_sum_assignment(-iou)
    return [(int(i), int(j), float(iou[i, j])) for i, j in zip(r, c) if iou[i, j] >= iou_thresh]
