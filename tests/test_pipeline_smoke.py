"""Day-0 integration test: the whole pipeline runs end-to-end on fixtures with the STUBS.
It must keep passing after every merge."""
import numpy as np
from docscan.config import load_config
from docscan.data.annotations import list_annotations, load_annotation, oracle_detections
from docscan.data.fixtures import make_fixtures
from docscan.geometry import match_quads
from docscan.io_utils import read_json
from docscan.pipeline import process_scene
from docscan.types import detection_from_dict


def test_fixture_and_oracle(tmp_path):
    root = make_fixtures(tmp_path / "fx", n=2)
    anns = list_annotations(root)
    assert len(anns) == 2
    ann = load_annotation(anns[0])
    gt_docs = [o for o in ann.objects if o.kind == "document"]
    dets = oracle_detections(ann)
    assert len(dets) == len(gt_docs) == 2
    m = match_quads([d.quad for d in dets], [o.quad for o in gt_docs], 0.99)
    assert len(m) == 2


def test_pipeline_runs_end_to_end(tmp_path):
    root = make_fixtures(tmp_path / "fx", n=1)
    ann = load_annotation(list_annotations(root)[0])
    cfg = load_config()
    res = process_scene(ann.image_path, cfg, tmp_path / "out")
    scene_dir = tmp_path / "out" / ann.scene_id
    assert (scene_dir / "detections.json").exists() and (scene_dir / "overlay.jpg").exists()
    assert res.n_accepted == len(res.docs) >= 1
    for rec in res.docs:
        for name in ("rectified", "gray") + tuple(f"bin_{m}" for m in cfg.binarize.methods):
            assert (scene_dir / rec.files[name]).exists()
    dets = [detection_from_dict(o) for o in read_json(scene_dir / "detections.json")]
    assert all(np.asarray(d.quad).shape == (4, 2) for d in dets)


def test_pipeline_with_oracle_detections(tmp_path):
    """Member B can run the back half with perfect detections, no detector needed."""
    root = make_fixtures(tmp_path / "fx", n=1)
    ann = load_annotation(list_annotations(root)[0])
    cfg = load_config()
    dets = oracle_detections(ann)
    res = process_scene(ann.image_path, cfg, tmp_path / "out", detections_override=dets)
    assert res.n_accepted == 2 and len(res.docs) == 2
