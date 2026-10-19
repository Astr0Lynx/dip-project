"""
docscan.data.fixtures -- tiny synthetic scenes for smoke tests. SHARED, FROZEN (v1.0).

Needs NO downloaded dataset. Produces the exact folder layout and JSON schema of the
real generator (see data/annotations.py), so both members can test against it on
day 0:

    python -m docscan.data.fixtures --out data/fixtures --n 3

Scenes are 800x600, tier T0 (no overlap, objects >= MIN_GAP_PX apart): 2 documents + 2 distractors each.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from ..geometry import quad_iou
from ..io_utils import ensure_dir, get_rng, write_image, write_json
from ..types import SceneAnnotation, SceneObject, annotation_to_dict

WORDS = ["invoice", "total", "date", "name", "address", "amount", "paid", "item", "tax",
         "receipt", "number", "street", "city", "card", "thank", "you", "balance", "due"]


def _text_page(rng, w: int, h: int) -> Tuple[np.ndarray, str]:
    """White page with several lines of dark text. Returns (image, text_rendered)."""
    img = np.full((h, w, 3), 245, np.uint8)
    cv2.rectangle(img, (2, 2), (w - 3, h - 3), (60, 60, 60), 2)
    lines, y = [], 28
    while y < h - 16:
        line = " ".join(rng.choice(WORDS, size=int(rng.integers(2, 5))))
        cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
        lines.append(line)
        y += 26
    return img, "\n".join(lines)


def _blank_page(rng, w: int, h: int) -> np.ndarray:
    img = np.full((h, w, 3), 240, np.uint8)
    noise = rng.normal(0, 2, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def _folder(rng, w: int, h: int) -> np.ndarray:
    color = tuple(int(c) for c in rng.integers(40, 200, 3))
    return np.full((h, w, 3), color, np.uint8)


MIN_GAP_PX = 10  # same margin as synthetic tier T0: touching objects merge into one contour


def _gap(qa: np.ndarray, qb: np.ndarray) -> float:
    """Boundary distance between two disjoint convex quads (attained at a vertex of one of them)."""
    da = min(abs(cv2.pointPolygonTest(qb.reshape(-1, 1, 2), (float(x), float(y)), True)) for x, y in qa)
    db = min(abs(cv2.pointPolygonTest(qa.reshape(-1, 1, 2), (float(x), float(y)), True)) for x, y in qb)
    return min(da, db)


def _place_quad(rng, w: int, h: int, canvas_wh: Tuple[int, int]) -> np.ndarray:
    """Rotated + slightly perspective-jittered quad for a w x h object (TL,TR,BR,BL of the OBJECT)."""
    cw, ch = canvas_wh
    ang = np.deg2rad(rng.uniform(-35, 35))
    cx, cy = rng.uniform(0.15 * cw, 0.85 * cw), rng.uniform(0.15 * ch, 0.85 * ch)
    corners = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]], np.float32)
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], np.float32)
    q = corners @ rot.T + np.array([cx, cy], np.float32)
    q += rng.uniform(-0.04, 0.04, q.shape).astype(np.float32) * np.array([w, h], np.float32)
    return q.astype(np.float32)


def make_scene(seed: int, scene_id: str, root: Path) -> SceneAnnotation:
    rng = get_rng(seed)
    cw, ch = 800, 600
    bg = np.full((ch, cw, 3), 0, np.uint8)
    base = rng.integers(90, 170, 3)
    bg[:] = base
    bg = np.clip(bg + rng.normal(0, 6, bg.shape), 0, 255).astype(np.uint8)
    grad = np.linspace(-20, 20, cw, dtype=np.float32)[None, :, None]
    scene = np.clip(bg.astype(np.float32) + grad, 0, 255).astype(np.uint8)

    specs = [("document", "note", (240, 160)), ("document", "receipt", (130, 260)),
             ("distractor", "blank_paper", (200, 150)), ("distractor", "folder", (170, 120))]
    objects: List[SceneObject] = []
    placed: List[np.ndarray] = []
    templates_dir = ensure_dir(root / "templates")

    for k, (kind, sub, (w, h)) in enumerate(specs):
        for _ in range(2000):  # rejection sampling: no overlaps in tier T0
            quad = _place_quad(rng, w, h, (cw, ch))
            inside = np.all((quad[:, 0] > 5) & (quad[:, 0] < cw - 5) & (quad[:, 1] > 5) & (quad[:, 1] < ch - 5))
            if inside and all(quad_iou(quad, p) == 0.0 and _gap(quad, p) >= MIN_GAP_PX for p in placed):
                break
        else:
            raise RuntimeError("could not place object; change seed")
        placed.append(quad)

        text = None
        tpl_rel = None
        if kind == "document":
            tpl, text = _text_page(rng, w, h)
            tpl_rel = f"templates/{scene_id}_obj{k}.png"
            write_image(root / tpl_rel, tpl)
        elif sub == "blank_paper":
            tpl = _blank_page(rng, w, h)
        else:
            tpl = _folder(rng, w, h)

        src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
        M = cv2.getPerspectiveTransform(src, quad)
        warped = cv2.warpPerspective(tpl, M, (cw, ch), flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (cw, ch), flags=cv2.INTER_NEAREST)
        scene[mask > 0] = warped[mask > 0]

        objects.append(SceneObject(obj_id=k, kind=kind, subtype=sub, quad=quad, z_order=k,
                                   visible_fraction=1.0, source="fixture",
                                   template_path=tpl_rel, text=text))

    img_rel = f"scenes/{scene_id}.jpg"
    write_image(root / img_rel, scene)
    ann = SceneAnnotation(scene_id=scene_id, image_path=img_rel, width=cw, height=ch,
                          tier="T0", seed=seed, objects=objects)
    write_json(root / "annotations" / f"{scene_id}.json", annotation_to_dict(ann))
    return ann


def make_fixtures(out_dir, n: int = 3, seed: int = 0) -> Path:
    root = ensure_dir(out_dir)
    ensure_dir(root / "scenes")
    ensure_dir(root / "annotations")
    for i in range(n):
        make_scene(seed + i, f"fx_{i:03d}", root)
    write_json(root / "manifest.json", {"version": "fixture-1", "n_scenes": n, "seed_offset": seed})
    return root


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/fixtures")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print("wrote", make_fixtures(a.out, a.n, a.seed))
