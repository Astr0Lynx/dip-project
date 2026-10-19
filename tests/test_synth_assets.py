"""Member A: source images for the generator (PLAN A1.2-A1.4) and the MIDV loader (A1.1)."""
import json

import cv2
import numpy as np
import pytest
from docscan.data.assets import (BackgroundPool, hash_split, load_wordlist, make_distractor,
                                 render_text_page)
from docscan.data.midv import index_midv, rectified_template, split_midv
from docscan.io_utils import get_rng
from docscan.types import DISTRACTOR_SUBTYPES


def test_wordlist_has_real_words():
    assert len(load_wordlist()) >= 300


@pytest.mark.parametrize("subtype,wh", [("note", (420, 600)), ("receipt", (220, 600))])
def test_render_text_page(subtype, wh):
    img, text = render_text_page(get_rng(1), wh, subtype)
    assert img.shape == (wh[1], wh[0], 3) and img.dtype == np.uint8
    lines = text.split("\n")
    assert len(lines) >= 6 and all(line.strip() for line in lines)
    # ink is present: a text page is clearly darker somewhere than blank paper
    assert (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) < 128).mean() > 0.002
    img2, text2 = render_text_page(get_rng(1), wh, subtype)
    assert text == text2 and np.array_equal(img, img2)


@pytest.mark.parametrize("subtype", DISTRACTOR_SUBTYPES)
def test_make_distractor(subtype):
    img = make_distractor(get_rng(2), subtype, (150, 210))
    assert img.shape == (210, 150, 3) and img.dtype == np.uint8


def test_background_pool_fallback_and_split(tmp_path):
    pool = BackgroundPool(tmp_path, "train")
    assert len(pool) == 0
    bg = pool.sample(get_rng(0), (320, 240))
    assert bg.shape == (240, 320, 3) and bg.dtype == np.uint8
    # files are split by path hash: every file lands in exactly one split
    d = tmp_path / "dtd" / "images" / "x"
    d.mkdir(parents=True)
    for i in range(30):
        cv2.imwrite(str(d / f"f{i}.jpg"), np.full((40, 50, 3), i * 8, np.uint8))
    ids = [BackgroundPool(tmp_path, s).ids for s in ("train", "val", "test")]
    assert sum(len(x) for x in ids) == 30 and len(set().union(*map(set, ids))) == 30
    assert hash_split("dtd/a.jpg") == hash_split("dtd/a.jpg")
    assert BackgroundPool(tmp_path, "train").sample(get_rng(1), (64, 48)).shape == (48, 64, 3)
    # desk backgrounds only from allowed DTD categories; textures still use every file
    other = tmp_path / "dtd" / "images" / "paisley"
    other.mkdir()
    for i in range(30):
        cv2.imwrite(str(other / f"p{i}.jpg"), np.full((40, 50, 3), 7, np.uint8))
    for s in ("train", "val", "test"):
        pool = BackgroundPool(tmp_path, s, categories=["x"])
        assert all(r.split("/")[2] == "x" for r in pool.desk_ids)
        assert len(pool.ids) >= len(pool.desk_ids)


def _fake_midv(root, doc_id, w=500, h=320):
    """Minimal folder in the expected MIDV-500 layout ([VERIFY] against the real download)."""
    d = root / doc_id
    (d / "images" / "TS").mkdir(parents=True)
    (d / "ground_truth" / "TS").mkdir(parents=True)
    tpl = np.full((h + 20, w + 20, 3), 30, np.uint8)
    tpl[10:10 + h, 10:10 + w] = 200
    cv2.imwrite(str(d / "images" / f"{doc_id}.tif"), tpl)
    quad = [[10, 10], [10 + w, 10], [10 + w, 10 + h], [10, 10 + h]]
    (d / "ground_truth" / f"{doc_id}.json").write_text(json.dumps({"quad": quad}))
    cv2.imwrite(str(d / "images" / "TS" / "TS01_01.tif"), tpl)
    (d / "ground_truth" / "TS" / "TS01_01.json").write_text(json.dumps({"quad": quad}))


def test_midv_index_template_and_split(tmp_path):
    for i in range(10):
        _fake_midv(tmp_path / "midv500", f"{i + 1:02d}_abc_id")
    docs = index_midv(tmp_path / "midv500")
    assert [d.doc_id for d in docs] == [f"{i + 1:02d}_abc_id" for i in range(10)]
    assert docs[0].template_image is not None and "TS" in docs[0].clips
    tpl = rectified_template(docs[0])
    assert 800 <= max(tpl.shape[:2]) <= 1000
    assert tpl.mean() > 150  # the dark border outside the GT quad was cut away
    parts = split_midv(docs)
    ids = [{d.doc_id for d in parts[s]} for s in ("train", "val", "test")]
    assert sum(map(len, ids)) == 10 and not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert parts == split_midv(docs)
    assert index_midv(tmp_path / "missing") == []
