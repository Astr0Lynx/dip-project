"""
docscan.detect.candidates -- Member A. Candidate quadrilaterals (PLAN A2).

    preprocess(scene, max_side)         -> (small_gray, scale)     scale = small / full
    edge_maps(small_gray, cfg)          -> [binary uint8 maps]     Canny, morph gradient, bright paper
    contour_candidates(edge_map, cfg)   -> [Candidate]             (downscaled coordinates)
    hough_candidates(edge_map, cfg)     -> [Candidate]             (downscaled coordinates)
    nms_candidates(cands, iou_thresh)   -> [Candidate]
    find_candidates(scene, cfg)         -> [Candidate]             PUBLIC, full-res, ordered, ids 0..n-1

Recall matters more than precision here: the rejection cascade (A4) removes distractors and
duplicates, but a document missed at this stage is lost for good.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..geometry import order_quad, quad_area, quad_iou
from ..types import Candidate

log = logging.getLogger("docscan.detect.candidates")

_SOURCE_RANK = {"contour": 0, "hough": 1, "overlap_resolved": 2}


# ----------------------------------------------------------------------------
# preprocessing and edge maps
# ----------------------------------------------------------------------------
def _downscale(scene: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    h, w = scene.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    if scale < 1.0:
        scene = cv2.resize(scene, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                           interpolation=cv2.INTER_AREA)
    return scene, scale


def preprocess(scene: np.ndarray, max_side: int, blur_ksize: int = 5) -> Tuple[np.ndarray, float]:
    """Downscale to `max_side` (never upscale), convert to gray, Gaussian blur."""
    small, scale = _downscale(scene, max_side)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    k = int(blur_ksize) | 1
    return (cv2.GaussianBlur(gray, (k, k), 0) if k > 1 else gray), scale


def _close(m: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return m
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))


def edge_maps(small_gray: np.ndarray, cfg, small_bgr: Optional[np.ndarray] = None) -> List[np.ndarray]:
    """Complementary binary maps (uint8 0/255), so no single lighting failure loses a page:
    (a) auto-Canny (capped thresholds) + closing, (a') strong-edge Canny + closing,
    (b) morphological gradient + adaptive threshold, (c) "bright paper" regions (low saturation,
    high value; only when `small_bgr` is given, always LAST)."""
    d = cfg.detect
    k = int(d.morph_close_ksize)
    med = float(np.median(small_gray))
    lo = min(max(0.0, (1 - d.canny_sigma) * med), float(d.canny_lo_max))
    hi = min((1 + d.canny_sigma) * med, float(d.canny_hi_max))
    maps = [_close(cv2.Canny(small_gray, lo, max(hi, lo + 1)), k)]
    # (a') strong edges only: page-vs-desk borders survive, most desk texture does not
    maps.append(_close(cv2.Canny(small_gray, float(d.canny_strong[0]), float(d.canny_strong[1])), k))

    grad = cv2.morphologyEx(small_gray, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    block = max(3, int(d.grad_block) | 1)
    maps.append(_close(cv2.adaptiveThreshold(grad, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                             block, float(d.grad_c)), k))

    if small_bgr is not None and small_bgr.ndim == 3:
        hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
        v = hsv[..., 2]
        otsu, _ = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        paper = ((hsv[..., 1] <= d.paper_s_max) & (v >= max(float(d.paper_v_min), otsu))).astype(np.uint8) * 255
        paper = cv2.morphologyEx(paper, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        maps.append(_close(paper, 2 * k + 1))  # wider closing fills text holes in the page
    return maps


# ----------------------------------------------------------------------------
# contour candidates
# ----------------------------------------------------------------------------
def _quad_ok(q: np.ndarray, img_area: float, cfg) -> bool:
    d = cfg.detect
    a = quad_area(q)
    if not (d.min_area_frac * img_area <= a <= d.max_area_frac * img_area):
        return False
    sides = np.linalg.norm(q - np.roll(q, -1, axis=0), axis=1)
    return bool(sides.min() >= d.min_side_px)


def _line_intersection(p1, p2, p3, p4) -> Optional[np.ndarray]:
    """Intersection of the infinite lines p1-p2 and p3-p4 (None if near parallel)."""
    d1, d2 = p2 - p1, p4 - p3
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-9 * max(1.0, float(np.dot(d1, d1) * np.dot(d2, d2)) ** 0.5):
        return None
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / den
    return p1 + t * d1


def _reduce_to_quad(poly: np.ndarray, max_growth: float = 1.25) -> Optional[np.ndarray]:
    """Collapse a convex 5-6-gon to a 4-gon by repeatedly removing its shortest side and
    extending the two neighbouring sides until they meet. Recovers a corner clipped by the
    image border, an occluding object or a rounded card corner. None if the result is not a
    convex quad or grows the area by more than `max_growth` (wild extrapolation)."""
    p = poly.reshape(-1, 2).astype(np.float64)
    a0 = float(cv2.contourArea(p.astype(np.float32)))
    while len(p) > 4:
        n = len(p)
        lens = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
        i = int(np.argmin(lens))  # side p[i] -> p[i+1]
        x = _line_intersection(p[(i - 1) % n], p[i], p[(i + 1) % n], p[(i + 2) % n])
        if x is None:
            return None
        p[i] = x
        p = np.delete(p, (i + 1) % n, axis=0)
    q = p.astype(np.float32)
    if not cv2.isContourConvex(q.reshape(-1, 1, 2)) or quad_area(q) > max_growth * max(a0, 1e-6):
        return None
    return q


def contour_candidates(edge_map: np.ndarray, cfg, retrieval: int = cv2.RETR_LIST) -> List[Candidate]:
    """Convex 4-gons from approxPolyDP (several epsilons; on the contour, then on its hull),
    plus a minAreaRect quad for large, mostly convex non-4-gons (merged / occluded pages)."""
    d = cfg.detect
    h, w = edge_map.shape[:2]
    img_area = float(h * w)
    contours, _ = cv2.findContours(edge_map, retrieval, cv2.CHAIN_APPROX_SIMPLE)
    out: List[Candidate] = []
    min_area, max_area = d.min_area_frac * img_area, d.max_area_frac * img_area
    for c in contours:
        if len(c) < 4:
            continue
        # filter on the HULL area: an outline with a gap (or fused with the page's text) is traced
        # as a thin open chain whose own area is tiny, but its hull is still the page
        x, y, bw, bh = cv2.boundingRect(c)
        if bw * bh < min_area:
            continue
        hull = cv2.convexHull(c)
        hull_area = float(cv2.contourArea(hull))
        if not (min_area <= hull_area <= max_area):
            continue
        area = float(cv2.contourArea(c))
        quad = None
        for poly in (c, hull):
            peri = cv2.arcLength(poly, True)
            for eps in d.approx_eps_fracs:
                ap = cv2.approxPolyDP(poly, eps * peri, True)
                if len(ap) == 4 and cv2.isContourConvex(ap):
                    q = ap.reshape(4, 2).astype(np.float32)
                    if _quad_ok(q, img_area, cfg):
                        quad = q
                        break
            if quad is not None:
                break
        if quad is None:
            # 5-6 sided hull: a page with one or two clipped corners
            peri = cv2.arcLength(hull, True)
            for eps in d.approx_eps_fracs:
                ap = cv2.approxPolyDP(hull, eps * peri, True)
                if 5 <= len(ap) <= 6:
                    q = _reduce_to_quad(ap)
                    if q is not None and _quad_ok(q, img_area, cfg):
                        quad = q
                        break
        if quad is None:
            # last resort: minAreaRect, only when the hull really is rectangle-like (an edge
            # contour's own area is meaningless, so compare the HULL with its bounding rectangle)
            rect = cv2.minAreaRect(hull)
            q = cv2.boxPoints(rect).astype(np.float32)
            if hull_area / max(quad_area(q), 1e-6) < d.min_fallback_solidity or not _quad_ok(q, img_area, cfg):
                continue
            quad = q
        qa = quad_area(quad)
        out.append(Candidate(cand_id=-1, quad=quad, contour=c.astype(np.int32), source="contour",
                             area=qa, quad_fit=float(np.clip(area / max(qa, 1e-6), 0.0, 1.0))))
    return out


# ----------------------------------------------------------------------------
# Hough fallback
# ----------------------------------------------------------------------------
def _angle_dist(a: np.ndarray, b: float, period: float) -> np.ndarray:
    d = np.abs((a - b) % period)
    return np.minimum(d, period - d)


def cluster_lines(lines: np.ndarray, tol_deg: float = 15.0,
                  max_families: int = 3) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Group segments (N,4) into pairs of near-perpendicular families.

    A rectangle puts BOTH of its side directions on the same value of (angle mod 90), so peaks
    of a length-weighted histogram of angle mod 90 give the dominant rectangle orientations
    (several documents may be rotated differently). Returns [(family_a, family_b), ...] as
    index arrays into `lines`, strongest orientation first."""
    if len(lines) == 0:
        return []
    dx, dy = lines[:, 2] - lines[:, 0], lines[:, 3] - lines[:, 1]
    ang = np.degrees(np.arctan2(dy, dx)) % 180.0
    length = np.hypot(dx, dy)
    hist = np.bincount(np.floor(ang % 90).astype(int) % 90, weights=length, minlength=90)
    win = max(1, int(tol_deg / 3))
    smooth = sum(np.roll(hist, s) for s in range(-win, win + 1))  # circular smoothing
    out = []
    used = np.zeros(len(lines), bool)
    for _ in range(max_families):
        if smooth.max() <= 0:
            break
        theta = float(np.argmax(smooth)) + 0.5
        fa = np.where((_angle_dist(ang, theta, 180.0) <= tol_deg) & ~used)[0]
        fb = np.where((_angle_dist(ang, theta + 90.0, 180.0) <= tol_deg) & ~used)[0]
        if len(fa) >= 2 and len(fb) >= 2:
            out.append((fa, fb))
        used[fa] = used[fb] = True
        smooth[_angle_dist(np.arange(90) + 0.5, theta, 90.0) <= tol_deg] = 0.0
    return out


def _homog(lines: np.ndarray) -> np.ndarray:
    p1 = np.c_[lines[:, 0:2], np.ones(len(lines))]
    p2 = np.c_[lines[:, 2:4], np.ones(len(lines))]
    return np.cross(p1, p2)


def quads_from_lines(a_lines: np.ndarray, b_lines: np.ndarray, shape: Tuple[int, int],
                     edge_map: Optional[np.ndarray] = None, n_samples: int = 16,
                     area_range: Tuple[float, float] = (0.0, np.inf), min_side: float = 0.0
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """All quads bounded by 2 lines of family A and 2 of family B (vectorised).
    Returns (quads (Q,4,2) float32 convex and inside the image, support (Q,) in [0,1] = fraction of
    points along the 4 sides that lie on `edge_map` (1.0 for all when no map is given))."""
    h, w = shape
    la, lb = _homog(a_lines.astype(np.float64)), _homog(b_lines.astype(np.float64))
    X = np.cross(la[:, None, :], lb[None, :, :])               # (nA, nB, 3) intersections
    with np.errstate(divide="ignore", invalid="ignore"):
        P = X[..., :2] / X[..., 2:3]
    ia, ja = np.triu_indices(len(la), 1)
    ib, jb = np.triu_indices(len(lb), 1)
    A1, A2 = np.repeat(ia, len(ib)), np.repeat(ja, len(ib))
    B1, B2 = np.tile(ib, len(ia)), np.tile(jb, len(ia))
    q = np.stack([P[A1, B1], P[A1, B2], P[A2, B2], P[A2, B1]], axis=1)  # cyclic corner order
    m = np.all(np.isfinite(q), axis=(1, 2))
    m &= np.all((q[..., 0] >= -0.02 * w) & (q[..., 0] <= 1.02 * w) &
                (q[..., 1] >= -0.02 * h) & (q[..., 1] <= 1.02 * h), axis=1)
    q = q[m]
    if len(q) == 0:
        return np.zeros((0, 4, 2), np.float32), np.zeros(0)
    e1, e2 = np.roll(q, -1, axis=1) - q, np.roll(q, -2, axis=1) - np.roll(q, -1, axis=1)
    cr = e1[..., 0] * e2[..., 1] - e1[..., 1] * e2[..., 0]
    q = q[np.all(cr > 0, axis=1) | np.all(cr < 0, axis=1)]
    # size filter BEFORE the (expensive) edge-support sampling
    x, y = q[..., 0], q[..., 1]
    area = 0.5 * np.abs((x * np.roll(y, -1, axis=1) - y * np.roll(x, -1, axis=1)).sum(axis=1))
    side = np.linalg.norm(q - np.roll(q, -1, axis=1), axis=2).min(axis=1)
    q = q[(area >= area_range[0]) & (area <= area_range[1]) & (side >= min_side)]
    if len(q) == 0 or edge_map is None:
        return q.astype(np.float32), np.ones(len(q))
    t = (np.arange(n_samples) + 0.5) / n_samples
    pts = q[:, :, None, :] + (np.roll(q, -1, axis=1) - q)[:, :, None, :] * t[None, None, :, None]
    xs = np.clip(np.round(pts[..., 0]).astype(int), 0, w - 1)
    ys = np.clip(np.round(pts[..., 1]).astype(int), 0, h - 1)
    support = (edge_map[ys, xs] > 0).reshape(len(q), -1).mean(axis=1)
    return q.astype(np.float32), support


def hough_candidates(edge_map: np.ndarray, cfg) -> List[Candidate]:
    """Quads from Hough line segments: recovers pages whose outline is broken (occlusion,
    low contrast) so no closed contour exists. Marked source="hough", contour=None, and
    quad_fit = edge support of the quad's sides (there is no contour to compare with)."""
    d, hc = cfg.detect, cfg.detect.hough
    h, w = edge_map.shape[:2]
    lines = cv2.HoughLinesP(edge_map, hc.rho, np.deg2rad(hc.theta_deg), int(hc.threshold),
                            minLineLength=hc.min_line_len_frac * max(h, w), maxLineGap=hc.max_line_gap_px)
    if lines is None:
        return []
    lines = lines.reshape(-1, 4).astype(np.float64)  # (N,1,4) in OpenCV 4, (N,4) in OpenCV 5
    length = np.hypot(lines[:, 2] - lines[:, 0], lines[:, 3] - lines[:, 1])
    support_map = cv2.dilate(edge_map, np.ones((3, 3), np.uint8))  # tolerate 1 px misfit
    img_area = float(h * w)
    quads, scores = [], []
    for fa, fb in cluster_lines(lines, hc.family_tol_deg, hc.max_families):
        keep = int(hc.max_lines_per_family)
        fa = fa[np.argsort(-length[fa])[:keep]]
        fb = fb[np.argsort(-length[fb])[:keep]]
        q, s = quads_from_lines(lines[fa], lines[fb], (h, w), support_map,
                                area_range=(d.min_area_frac * img_area, d.max_area_frac * img_area),
                                min_side=float(d.min_side_px))
        ok = s >= hc.min_support
        quads.append(q[ok])
        scores.append(s[ok])
    if not quads:
        return []
    q, s = np.concatenate(quads), np.concatenate(scores)
    out: List[Candidate] = []
    # greedy NMS over the best-supported quads only (most of the rest are near-duplicates)
    for i in np.argsort(-s, kind="stable")[: 20 * int(hc.max_quads)]:
        qi = q[i]
        if any(quad_iou(qi, c.quad) > d.nms_iou for c in out):
            continue
        out.append(Candidate(cand_id=-1, quad=qi, contour=None, source="hough",
                             area=quad_area(qi), quad_fit=float(np.clip(s[i], 0.0, 1.0))))
        if len(out) >= int(hc.max_quads):
            break
    return out


# ----------------------------------------------------------------------------
# merge and public entry point
# ----------------------------------------------------------------------------
def nms_candidates(cands: Sequence[Candidate], iou_thresh: float) -> List[Candidate]:
    """Greedy suppression of near-duplicates (IoU > iou_thresh). Preference order: contour
    before Hough (exact outline), then higher quad_fit, then larger area."""
    order = sorted(cands, key=lambda c: (_SOURCE_RANK.get(c.source, 9), -round(c.quad_fit, 2), -c.area))
    kept: List[Candidate] = []
    for c in order:
        if all(quad_iou(c.quad, k.quad) <= iou_thresh for k in kept):
            kept.append(c)
    return kept


def find_candidates(scene: np.ndarray, cfg) -> List[Candidate]:
    """All candidate document quads of a FULL-RES scene: ordered float32 quads and int32
    contours in full-res coordinates, cand_id 0..n-1. Never raises on odd input ([] instead)."""
    try:
        if scene is None or scene.ndim not in (2, 3) or min(scene.shape[:2]) < 8:
            return []
        if scene.ndim == 2:
            scene = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
        d = cfg.detect
        small, scale = _downscale(scene, int(d.max_side))
        gray, _ = preprocess(small, int(d.max_side), int(d.blur_ksize))
        maps = edge_maps(gray, cfg, small_bgr=small)
        cands: List[Candidate] = []
        for i, m in enumerate(maps):
            is_region_map = i == 3  # the paper map (last) holds filled regions, not outlines
            cs = contour_candidates(m, cfg, cv2.RETR_EXTERNAL if is_region_map else cv2.RETR_LIST)
            cands += cs
            # Hough only on the thin Canny maps: the gradient map is thick and, on textured
            # desks, mostly texture (many lines, few page sides). Measured on 80 val scenes:
            # adding it gives T1 recall 0.72 -> 0.80 but 2.4x the time (max 2.7 s > budget)
            if d.use_hough_fallback and i < 2 and len(cs) < int(d.hough.run_if_fewer_than):
                cands += hough_candidates(m, cfg)
        cands = nms_candidates(cands, float(d.nms_iou))
        # cap by outline quality first, then area: an area-only cap fills up with big texture
        # quads on cluttered desks and drops small documents (ID cards)
        cands = sorted(cands, key=lambda c: (_SOURCE_RANK.get(c.source, 9), -round(c.quad_fit, 1),
                                             -c.area))[: int(d.max_candidates)]
        cands = sorted(cands, key=lambda c: -c.area)
        out: List[Candidate] = []
        for c in cands:
            q = order_quad(c.quad / np.float32(scale))
            contour = None
            if c.contour is not None:
                contour = np.round(c.contour.astype(np.float64) / scale).astype(np.int32).reshape(-1, 1, 2)
            out.append(Candidate(cand_id=len(out), quad=q, contour=contour, source=c.source,
                                 area=float(quad_area(q)), quad_fit=float(c.quad_fit)))
        return out
    except (cv2.error, ValueError, IndexError) as e:  # a stage never crashes the pipeline
        log.warning("find_candidates failed (%s); returning no candidates", e)
        return []
