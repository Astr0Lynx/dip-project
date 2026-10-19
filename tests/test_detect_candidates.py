"""Member A: candidate detection (PLAN A2 tests)."""
import time

import numpy as np
import pytest
from docscan.config import load_config
from docscan.data.annotations import list_annotations, load_annotation
from docscan.data.fixtures import make_fixtures
from docscan.detect.api import find_candidates
from docscan.detect.candidates import cluster_lines, hough_candidates, nms_candidates
from docscan.geometry import match_quads, order_quad
from docscan.io_utils import get_rng, read_image
from docscan.types import Candidate, Detection
from docscan.validate import validate_detections


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def fixture_root(tmp_path_factory):
    return make_fixtures(tmp_path_factory.mktemp("fx"), n=3)


def test_every_fixture_document_has_a_candidate(cfg, fixture_root):
    for p in list_annotations(fixture_root):
        ann = load_annotation(p)
        cands = find_candidates(read_image(ann.image_path), cfg)
        docs = [o.quad for o in ann.objects if o.kind == "document"]
        assert len(match_quads([c.quad for c in cands], docs, 0.8)) == len(docs)


def test_outputs_are_full_res_ordered_and_valid(cfg, fixture_root):
    ann = load_annotation(list_annotations(fixture_root)[0])
    scene = read_image(ann.image_path)
    cands = find_candidates(scene, cfg)
    assert [c.cand_id for c in cands] == list(range(len(cands)))
    for c in cands:
        assert c.quad.dtype == np.float32 and c.quad.shape == (4, 2)
        assert np.allclose(c.quad, order_quad(c.quad))
        assert isinstance(c.area, float) and isinstance(c.quad_fit, float) and 0 <= c.quad_fit <= 1
        if c.contour is not None:
            assert c.contour.dtype == np.int32 and c.contour.shape[1:] == (1, 2)
    dets = [Detection(det_id=c.cand_id, quad=c.quad, score=1.0, accepted=True, decided_by="heuristic",
                      source=c.source, contour=c.contour, quad_fit=c.quad_fit) for c in cands]
    validate_detections(dets, scene.shape[:2])


def test_coordinates_are_full_resolution(cfg, fixture_root):
    """A 2x upscaled scene (above detect.max_side) must give 2x quads."""
    import cv2
    ann = load_annotation(list_annotations(fixture_root)[0])
    scene = read_image(ann.image_path)
    big = cv2.resize(scene, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    docs = [o.quad * 2 for o in ann.objects if o.kind == "document"]
    cands = find_candidates(big, cfg)
    assert len(match_quads([c.quad for c in cands], docs, 0.8)) == len(docs)


@pytest.mark.parametrize("img", [
    np.full((600, 800, 3), 128, np.uint8),
    get_rng(0).integers(0, 256, (600, 800, 3)).astype(np.uint8),
    np.zeros((1, 1, 3), np.uint8),
    np.zeros((600, 800), np.uint8),
])
def test_no_crash_on_odd_images(cfg, img):
    out = find_candidates(img, cfg)
    assert isinstance(out, list)
    if img.size <= 3 or img.std() == 0:
        assert out == []


def test_runtime_full_size_scene(cfg, fixture_root):
    import cv2
    ann = load_annotation(list_annotations(fixture_root)[0])
    scene = cv2.resize(read_image(ann.image_path), (1600, 1200))
    find_candidates(scene, cfg)  # warm-up (imports, caches)
    t = time.perf_counter()
    find_candidates(scene, cfg)
    assert time.perf_counter() - t < 1.5


def test_hough_recovers_a_page_with_broken_outline(cfg):
    """Outline drawn as 4 separate segments with gaps: no closed contour, Hough must find it."""
    import cv2
    m = np.zeros((600, 800), np.uint8)
    q = np.array([[200, 150], [600, 170], [580, 450], [190, 430]], np.float32)
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        cv2.line(m, tuple(int(v) for v in a + 0.15 * (b - a)), tuple(int(v) for v in a + 0.85 * (b - a)), 255, 1)
    cands = hough_candidates(m, cfg)
    assert cands and all(c.source == "hough" and c.contour is None for c in cands)
    assert match_quads([c.quad for c in cands], [q], 0.8)


def test_page_with_clipped_corner_is_recovered(cfg):
    """A page partly outside the frame shows a 5-sided outline: its corner is rebuilt."""
    import cv2
    from docscan.detect.candidates import _reduce_to_quad, contour_candidates
    q = np.array([[-30, 100], [300, 60], [340, 330], [10, 380]], np.float32)  # TL corner outside
    poly = np.array([[0, 96], [300, 60], [340, 330], [10, 380], [0, 230]], np.float32)  # clipped view
    rq = _reduce_to_quad(poly)
    assert rq is not None and match_quads([rq], [q], 0.95)
    m = np.zeros((450, 600), np.uint8)
    cv2.polylines(m, [np.round(poly).astype(np.int32).reshape(-1, 1, 2)], True, 255, 1)
    assert match_quads([c.quad for c in contour_candidates(m, cfg)], [q], 0.9)


def test_cluster_lines_separates_orientations():
    lines = np.array([[0, 0, 100, 0], [0, 50, 100, 52], [0, 0, 0, 100], [50, 0, 52, 100],
                      [0, 0, 70, 70], [10, 0, 80, 70], [0, 70, 70, 0], [0, 80, 70, 10]], np.float64)
    fams = cluster_lines(lines, tol_deg=10, max_families=3)
    groups = sorted(sorted(map(int, a)) + sorted(map(int, b)) for a, b in fams)
    assert groups == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_nms_keeps_one_of_duplicates():
    q = order_quad([[0, 0], [100, 0], [100, 80], [0, 80]])
    c1 = Candidate(0, q, None, "hough", 8000.0, 0.7)
    c2 = Candidate(1, q + 1, None, "contour", 8000.0, 0.95)
    c3 = Candidate(2, q + 300, None, "contour", 8000.0, 0.9)
    kept = nms_candidates([c1, c2, c3], 0.7)
    assert [c.cand_id for c in kept] == [1, 2]  # contour beats hough for the duplicate pair
