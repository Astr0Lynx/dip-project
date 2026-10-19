"""Contract v1.1: how pipeline.process_scene drives the overlap stage, plus its new options."""
import numpy as np
import docscan.pipeline as pipeline
from docscan.config import load_config
from docscan.data.annotations import list_annotations, load_annotation, oracle_detections
from docscan.detect.api import classify_candidates
from docscan.geometry import match_quads, scale_quad
from docscan.io_utils import read_image, read_json
from docscan.overlap.resolve import resolve_overlaps
from docscan.types import Detection


def _fixture(tmp_path):
    from docscan.data.fixtures import make_fixtures
    root = make_fixtures(tmp_path / "fx", n=1)
    return load_annotation(list_annotations(root)[0])


def _with_rejected(ann):
    """Oracle documents (det 0, 1 accepted) plus one rejected candidate (det 2), A's id layout."""
    dets = oracle_detections(ann)
    far = np.array([[600, 450], [700, 450], [700, 550], [600, 550]], np.float32)
    return dets + [Detection(det_id=2, quad=far, score=0.1, accepted=False,
                             decided_by="ml", source="contour")]


def _split_first_accepted(scene, dets, cfg, classify=None):
    """Fake v1.1 overlap stage: replace det 0 by two halves with ids from max(ALL ids) + 1."""
    _split_first_accepted.seen = ([d.det_id for d in dets], classify)
    nxt = max(d.det_id for d in dets) + 1
    parent = next(d for d in dets if d.accepted)
    tl, tr, br, bl = parent.quad
    mt, mb = (tl + tr) / 2, (bl + br) / 2
    kids = [Detection(det_id=nxt + i, quad=q.astype(np.float32), score=parent.score, accepted=True,
                      decided_by=parent.decided_by, source="overlap_resolved")
            for i, q in enumerate([np.array([tl, mt, mb, bl]), np.array([mt, tr, br, mb])])]
    return [d for d in dets if d.det_id != parent.det_id] + kids


def test_stub_is_identity_over_all_detections(tmp_path):
    ann = _fixture(tmp_path)
    dets = _with_rejected(ann)
    out = resolve_overlaps(read_image(ann.image_path), dets, load_config(), classify=None)
    assert [d.det_id for d in out] == [0, 1, 2] and [d.accepted for d in out] == [True, True, False]


def test_overlap_stage_sees_all_and_final_list_is_written(tmp_path, monkeypatch):
    ann = _fixture(tmp_path)
    monkeypatch.setattr(pipeline, "resolve_overlaps", _split_first_accepted)
    cfg = load_config(overrides={"overlap.enabled": True})
    res = pipeline.process_scene(ann.image_path, cfg, tmp_path / "out",
                                 detections_override=_with_rejected(ann))
    seen_ids, classify = _split_first_accepted.seen
    assert seen_ids == [0, 1, 2] and classify is None          # ALL in; oracle -> no classifier
    d = tmp_path / "out" / ann.scene_id
    final = read_json(d / "detections.json")
    assert sorted(o["det_id"] for o in final) == [1, 2, 3, 4]  # parent gone, kids added, no collision
    assert {o["source"] for o in final if o["det_id"] >= 3} == {"overlap_resolved"}
    assert sorted(o["det_id"] for o in read_json(d / "detections_raw.json")) == [0, 1, 2]
    assert sorted(r.det_id for r in res.docs) == [1, 3, 4] and res.n_accepted == 3
    for rec in res.docs:                                        # files are named by det_id
        assert rec.files["rectified"] == f"doc_{rec.det_id:02d}_rectified.png"
        assert (d / rec.files["rectified"]).exists()


def test_real_detector_path_injects_classifier(tmp_path, monkeypatch):
    ann = _fixture(tmp_path)
    monkeypatch.setattr(pipeline, "resolve_overlaps", _split_first_accepted)
    cfg = load_config(overrides={"overlap.enabled": True})
    pipeline.process_scene(ann.image_path, cfg, tmp_path / "out")
    assert _split_first_accepted.seen[1] is classify_candidates


def test_on_doc_callback_without_images(tmp_path):
    ann = _fixture(tmp_path)
    cfg = load_config()
    seen = []
    res = pipeline.process_scene(
        ann.image_path, cfg, tmp_path / "out", detections_override=oracle_detections(ann),
        save_images=False, on_doc=lambda det, rect, enh, bins: seen.append(
            (det.det_id, rect.image.shape[:2] == enh.gray.shape, sorted(bins))))
    d = tmp_path / "out" / ann.scene_id
    assert not list(d.glob("*.png")) and not list(d.glob("*.jpg"))
    assert (d / "detections.json").exists() and (d / "result.json").exists()
    assert seen == [(0, True, sorted(cfg.binarize.methods)), (1, True, sorted(cfg.binarize.methods))]
    assert "on_doc" in res.timings_s and all(rec.files == {} for rec in res.docs)


def test_large_scene_is_downscaled_and_override_rescaled(tmp_path):
    ann = _fixture(tmp_path)                                    # 800 x 600
    cfg = load_config(overrides={"pipeline.max_scene_side": 400})
    pipeline.process_scene(ann.image_path, cfg, tmp_path / "out",
                           detections_override=oracle_detections(ann))
    d = tmp_path / "out" / ann.scene_id
    assert read_json(d / "result.json")["scene_scale"] == [0.5, 0.5]
    pred = [np.asarray(o["quad"], np.float32) for o in read_json(d / "detections.json")]
    gt = [scale_quad(o.quad, 0.5) for o in ann.objects if o.kind == "document"]
    assert len(match_quads(pred, gt, iou_thresh=0.99)) == 2
