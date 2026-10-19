import numpy as np
from docscan.geometry import (best_cyclic_alignment, is_convex_quad, match_quads, order_quad,
                              quad_area, quad_iou, quad_target_size)


def rect(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)


def test_order_quad_is_shuffle_invariant():
    q = rect(10, 20, 110, 70)
    rng = np.random.default_rng(0)
    for _ in range(10):
        shuffled = q[rng.permutation(4)]
        assert np.allclose(order_quad(shuffled), q)


def test_order_quad_rotated_is_clockwise_from_topleftmost():
    ang = np.deg2rad(30)
    r = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], np.float32)
    q = (rect(-50, -30, 50, 30) @ r.T + 200).astype(np.float32)
    o = order_quad(q)
    assert o[0].sum() == min(o.sum(axis=1))
    assert is_convex_quad(o)
    assert abs(quad_area(o) - 100 * 60) < 1e-2


def test_iou_identity_disjoint_half():
    a = rect(0, 0, 100, 100)
    assert abs(quad_iou(a, a) - 1.0) < 1e-4
    assert quad_iou(a, rect(200, 200, 300, 300)) == 0.0
    assert abs(quad_iou(a, rect(50, 0, 150, 100)) - (5000 / 15000)) < 1e-3


def test_cyclic_alignment_and_matching():
    gt = rect(0, 0, 100, 60)
    pred = np.roll(gt, 2, axis=0)
    aligned, d = best_cyclic_alignment(pred, gt)
    assert d < 1e-4 and np.allclose(aligned, gt)
    m = match_quads([rect(500, 500, 600, 560), pred], [gt])
    assert m == [(1, 0, m[0][2])] and m[0][2] > 0.99


def test_target_size_bounds():
    w, h = quad_target_size(rect(0, 0, 4000, 2000), max_side=1600, min_side=64)
    assert max(w, h) == 1600 and h == 800
    w, h = quad_target_size(rect(0, 0, 10, 10), max_side=1600, min_side=64)
    assert min(w, h) == 64
