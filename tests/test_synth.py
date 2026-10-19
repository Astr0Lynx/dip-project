"""Member A: synthetic scene generator (PLAN A1 tests)."""
import json
import os

import numpy as np
import pytest
from docscan.config import load_config
from docscan.data.annotations import list_annotations, load_annotation
from docscan.data.synth import (build_split, generate_scene, inside_frac, make_pools, overlap_ratio)
from docscan.geometry import is_convex_quad
from docscan.io_utils import get_rng, read_json
from docscan.types import annotation_from_dict, annotation_to_dict

# small canvas + small templates: same code paths, fast enough for 100+ scenes in the test budget
SMALL = {"synth.canvas_wh": [480, 360], "synth.templates.long_side": [200, 260]}


@pytest.fixture(scope="module")
def cfg():
    return load_config(overrides=SMALL)


@pytest.fixture(scope="module")
def pools(cfg, tmp_path_factory):
    # empty raw root: procedural backgrounds, no MIDV (tests run without downloads)
    return make_pools(cfg, "val", raw_root=tmp_path_factory.mktemp("raw"))


def _gen(cfg, pools, seed, tier):
    sink = {}
    scene, ann = generate_scene(get_rng(seed), cfg, pools, tier, scene_id=f"s{seed}", seed=seed,
                                template_sink=sink)
    return scene, ann, sink


def test_deterministic(cfg, pools):
    s1, a1, _ = _gen(cfg, pools, 7, "T1")
    s2, a2, _ = _gen(cfg, pools, 7, "T1")
    assert json.dumps(annotation_to_dict(a1)) == json.dumps(annotation_to_dict(a2))
    assert np.array_equal(s1, s2)


@pytest.mark.parametrize("tier", ["T0", "T1", "T2"])
def test_tier_constraints_hold(cfg, pools, tier):
    lo, hi = cfg.synth.overlap_limits.to_dict()[tier]
    w, h = cfg.synth.canvas_wh
    for seed in range(35):
        scene, ann, _ = _gen(cfg, pools, 1000 + seed, tier)
        assert scene.shape == (h, w, 3) and scene.dtype == np.uint8
        objs = ann.objects
        for o in objs:
            assert is_convex_quad(o.quad)
            assert inside_frac(o.quad, w, h) >= cfg.synth.min_inside_frac - 1e-6
            assert 0.0 <= o.visible_fraction <= 1.0
        ratios = {(i, j): overlap_ratio(objs[i].quad, objs[j].quad)
                  for i in range(len(objs)) for j in range(i + 1, len(objs))}
        if tier == "T0":
            assert all(r == 0.0 for r in ratios.values())
            assert all(o.visible_fraction == 1.0 for o in objs)
        else:
            assert len(objs) >= 2
            assert all(r <= hi + 1e-6 for r in ratios.values())
            # v1.1: the qualifying pair involves at least one document
            assert any(lo <= r <= hi and r > 0 and "document" in (objs[i].kind, objs[j].kind)
                       for (i, j), r in ratios.items())


def test_annotation_round_trip_and_text(cfg, pools):
    _, ann, sink = _gen(cfg, pools, 3, "T0")
    d = annotation_to_dict(ann)
    assert annotation_to_dict(annotation_from_dict(json.loads(json.dumps(d)))) == d
    for o in ann.objects:
        if o.subtype in ("note", "receipt"):
            assert o.text and o.text.strip()
            assert o.template_path in sink
        if o.kind == "distractor":
            assert o.template_path is None and o.text is None
    assert any(o.kind == "document" for o in ann.objects)


def test_build_split_layout_resumable(cfg, tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    root = build_split("val", cfg, tmp_path / "val", n=3, raw_root=raw, progress=False)
    paths = list_annotations(root)
    assert len(paths) == 3
    man = read_json(root / "manifest.json")
    assert man["version"] == "synth_v1" and man["n_scenes"] == 3 and man["seed_offset"] == 10000
    for p in paths:
        ann = load_annotation(p)
        assert ann.scene_id == p.stem and ann.seed >= 10000
        assert (root / "scenes" / f"{ann.scene_id}.jpg").exists()
        for o in ann.objects:
            if o.template_path:
                assert os.path.exists(o.template_path)
    before = {p: p.stat().st_mtime_ns for p in paths}
    build_split("val", cfg, root, n=3, raw_root=raw, progress=False)  # resumable: nothing rewritten
    assert {p: p.stat().st_mtime_ns for p in paths} == before
