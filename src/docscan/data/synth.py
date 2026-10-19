"""
docscan.data.synth -- Member A. Synthetic multi-document scene generator (PLAN A1.5, A1.6).

PUBLIC CONTRACT:
    generate_scene(rng, cfg, pools, tier) -> (scene_bgr uint8, SceneAnnotation)
    build_split(split, cfg, out_root, n=None) -> Path   # writes the layout in data/annotations.py
    make_pools(cfg, split)                -> ScenePools  # sources of ONE split (no leakage)
CLI:
    python -m docscan.data.synth --split train --out data/synth/train [--n 50] [--workers 4]
    python -m docscan.data.synth --contact-sheet data/synth/val --out results/contact_val.jpg

Camera model (O1, adopted): objects are laid out as exact rectangles on the desk plane and the
whole desk (background included) is seen through ONE random homography, as in a real photo.
Tier constraints (overlap ratios, 95% inside, T0 margin) are checked on the final IMAGE quads,
so the annotation always satisfies them exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..config import load_config, resolve_path
from ..geometry import is_convex_quad, quad_area, quad_bbox, quad_edge_lengths
from ..io_utils import ensure_dir, get_rng, write_json
from ..types import CONTRACT_VERSION, TIERS, SceneAnnotation, SceneObject, annotation_to_dict
from .assets import BackgroundPool, make_distractor_rgba, render_text_page
from .midv import MidvDoc, index_midv, rectified_template, split_midv

log = logging.getLogger("docscan.data.synth")

DATASET_VERSION = "synth_v1"

# width / height ranges of distractor sources (portrait; rotation makes some landscape)
_DISTRACTOR_ASPECT = {"blank_paper": (0.68, 0.78), "folder": (0.70, 0.85), "book": (0.62, 0.80),
                      "photo": (0.65, 0.80), "device": (0.45, 0.78)}


# ----------------------------------------------------------------------------
# pools: every source a scene of one split may use
# ----------------------------------------------------------------------------
@dataclass
class ScenePools:
    split: str
    backgrounds: BackgroundPool
    midv_docs: List[MidvDoc]
    _templates: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def midv_template(self, doc: MidvDoc) -> np.ndarray:
        """Exact template that is composited AND stored (cached: loaded once per process)."""
        if doc.doc_id not in self._templates:
            self._templates[doc.doc_id] = rectified_template(doc)
        return self._templates[doc.doc_id]


def make_pools(cfg, split: str, raw_root=None) -> ScenePools:
    """Backgrounds and MIDV document types of `split` (70/15/15 by source identity, seed 42)."""
    s = cfg.synth
    raw_root = Path(raw_root) if raw_root is not None else resolve_path(cfg.paths.data_root) / "raw"
    fracs, seed = tuple(s.split_fracs), int(s.split_seed)
    docs = split_midv(index_midv(raw_root / "midv500"), fracs, seed)[split] if (raw_root / "midv500").exists() else []
    bg = BackgroundPool(raw_root, split, fracs, seed, categories=list(s.background_categories),
                        procedural_prob=float(s.procedural_background_prob))
    pools = ScenePools(split=split, backgrounds=bg, midv_docs=[])
    for d in docs:  # keep only document types whose template can actually be built
        try:
            pools.midv_template(d)
            pools.midv_docs.append(d)
        except (FileNotFoundError, cv2.error) as e:
            log.warning("skipping MIDV %s: %s", d.doc_id, e)
    if not pools.midv_docs:
        log.warning("no MIDV templates for split %s: `id` documents are disabled", split)
    return pools


# ----------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------
def _inter_area(a: np.ndarray, b: np.ndarray) -> float:
    inter, _ = cv2.intersectConvexConvex(np.ascontiguousarray(a, np.float32),
                                         np.ascontiguousarray(b, np.float32))
    return float(inter)


def overlap_ratio(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection area / area of the SMALLER quad (the tier measure, PLAN 5.3)."""
    small = min(quad_area(a), quad_area(b))
    return _inter_area(a, b) / small if small > 1e-6 else 0.0


def _gap(a: np.ndarray, b: np.ndarray) -> float:
    """Distance between two disjoint convex quads (attained at a vertex of one of them)."""
    da = min(abs(cv2.pointPolygonTest(b.reshape(-1, 1, 2), (float(x), float(y)), True)) for x, y in a)
    db = min(abs(cv2.pointPolygonTest(a.reshape(-1, 1, 2), (float(x), float(y)), True)) for x, y in b)
    return float(min(da, db))


def _canvas_quad(w: int, h: int) -> np.ndarray:
    return np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)


def inside_frac(q: np.ndarray, w: int, h: int) -> float:
    """Fraction of the quad's area inside the canvas."""
    a = quad_area(q)
    return _inter_area(q, _canvas_quad(w, h)) / a if a > 1e-6 else 0.0


def _camera(rng: np.random.Generator, w: int, h: int, jitter: float) -> np.ndarray:
    """ONE homography desk -> image per scene (O1). The desk-plane preimages of the 4 image
    corners are jittered by up to `jitter` x canvas side; a single projective map keeps every
    object, and the background, in the same perspective."""
    img_c = _canvas_quad(w, h)
    desk_c = img_c + rng.uniform(-jitter, jitter, (4, 2)).astype(np.float32) * np.array([w, h], np.float32)
    return cv2.getPerspectiveTransform(desk_c, img_c)


def _to_image(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2).astype(np.float32)


def _desk_rect(c: np.ndarray, wh: Tuple[float, float], ang_deg: float) -> np.ndarray:
    """Rectangle on the desk in the object's OWN corner order (TL,TR,BR,BL of the source)."""
    w, h = wh
    a = np.deg2rad(ang_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    corners = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    return (corners @ rot.T + c).astype(np.float32)


# ----------------------------------------------------------------------------
# layout (rejection sampling, PLAN A1.5 step 4)
# ----------------------------------------------------------------------------
def _layout(rng: np.random.Generator, s, sizes: List[Tuple[float, float]], angles: List[float],
            tier: str, H: np.ndarray, desk_box: Tuple[float, float, float, float],
            canvas_wh: Tuple[int, int]) -> Optional[List[np.ndarray]]:
    """Image quads for every object, or None if some object could not be placed.

    T0: no intersection and at least `t0_margin_px` apart. T1/T2: object 1 is placed against
    object 0 (always a document) with an overlap ratio inside the tier range: the qualifying
    pair. Every other pair stays at or below the tier maximum.
    """
    cw, ch = canvas_wh
    lo, hi = (float(v) for v in getattr(s.overlap_limits, tier))
    x0, y0, x1, y1 = desk_box
    quads: List[np.ndarray] = []
    centers: List[np.ndarray] = []
    for k, (wh, ang) in enumerate(zip(sizes, angles)):
        placed = False
        for _ in range(int(s.max_place_tries)):
            if tier != "T0" and k == 1:
                # partner of the qualifying pair: centre near object 0 so they overlap
                reach = 0.5 * (np.hypot(*sizes[0]) + np.hypot(*wh))
                r, phi = reach * np.sqrt(rng.random()), rng.uniform(0, 2 * np.pi)
                c = centers[0] + r * np.array([np.cos(phi), np.sin(phi)])
            else:
                c = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
            q = _to_image(H, _desk_rect(c, wh, ang))
            if not is_convex_quad(q) or inside_frac(q, cw, ch) < float(s.min_inside_frac):
                continue
            if tier == "T0":
                ok = all(_inter_area(q, p) <= 0.0 and _gap(q, p) >= float(s.t0_margin_px) for p in quads)
            else:
                ratios = [overlap_ratio(q, p) for p in quads]
                ok = all(r <= hi for r in ratios)
                if ok and k == 1:
                    ok = lo <= ratios[0] <= hi and ratios[0] > 0.0
            if ok:
                quads.append(q)
                centers.append(c)
                placed = True
                break
        if not placed:
            return None
    return quads


# ----------------------------------------------------------------------------
# compositing
# ----------------------------------------------------------------------------
def _warp_object(img: np.ndarray, alpha: Optional[np.ndarray], quad: np.ndarray,
                 canvas_wh: Tuple[int, int], pad: int):
    """Warp a source image onto `quad` inside a padded ROI. Returns (x0, y0, warped f32, alpha f32)."""
    cw, ch = canvas_wh
    hs, ws = img.shape[:2]
    e = quad_edge_lengths(quad)
    # shrink with INTER_AREA first: a bilinear warp alone aliases small text badly
    s = min(1.0, 1.25 * max(max(e[0], e[2]) / ws, max(e[1], e[3]) / hs))
    if s < 1.0:
        size = (max(2, int(round(ws * s))), max(2, int(round(hs * s))))
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        if alpha is not None:
            alpha = cv2.resize(alpha, size, interpolation=cv2.INTER_AREA)
        hs, ws = img.shape[:2]
    bx0, by0, bx1, by1 = quad_bbox(quad)
    x0, y0 = max(0, bx0 - pad), max(0, by0 - pad)
    x1, y1 = min(cw, bx1 + pad), min(ch, by1 + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    # pixel-edge convention: the source's outer border lands exactly on the annotated quad
    src = np.array([[-0.5, -0.5], [ws - 0.5, -0.5], [ws - 0.5, hs - 0.5], [-0.5, hs - 0.5]], np.float32)
    M = cv2.getPerspectiveTransform(src, (quad - np.array([x0, y0], np.float32)).astype(np.float32))
    size = (x1 - x0, y1 - y0)
    warped = cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    a_src = alpha if alpha is not None else np.full((hs, ws), 255, np.uint8)
    a = cv2.warpPerspective(a_src, M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0)  # bilinear border = anti-aliased edge
    return x0, y0, warped.astype(np.float32), a.astype(np.float32) / 255.0


def _blur_sigma(img: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), sigmaX=max(sigma, 0.1)) if sigma > 0.05 else img


def _photometric(rng: np.random.Generator, s, scene: np.ndarray) -> np.ndarray:
    """Global light gradient, cast shadow, glare, blur and noise (PLAN A1.5 step 6)."""
    h, w = scene.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    phi = rng.uniform(0, 2 * np.pi)
    t = (np.cos(phi) * xx / w + np.sin(phi) * yy / h)
    t = (t - t.min()) / max(float(t.max() - t.min()), 1e-6)
    gmin = rng.uniform(*s.light_gradient_min)
    scene *= (gmin + (1.0 - gmin) * t)[..., None]

    if rng.random() < float(s.shadow_prob):
        # a soft cast shadow (hand, lamp arm): blurred random polygon, multiplied in
        c = np.array([rng.uniform(0, w), rng.uniform(0, h)])
        n = int(rng.integers(3, 7))
        ang = np.sort(rng.uniform(0, 2 * np.pi, n))
        rad = rng.uniform(0.2, 0.6, n) * max(w, h)
        poly = (c + np.stack([np.cos(ang) * rad, np.sin(ang) * rad], 1)).astype(np.int32)
        m = np.zeros((h, w), np.float32)
        cv2.fillPoly(m, [poly.reshape(-1, 1, 2)], 1.0)
        m = _blur_sigma(m, rng.uniform(0.02, 0.06) * max(w, h))
        scene *= (1.0 - (1.0 - rng.uniform(*s.shadow_strength)) * m)[..., None]

    if rng.random() < float(s.glare_prob):
        m = np.zeros((h, w), np.float32)
        axes = (int(rng.uniform(0.05, 0.2) * w), int(rng.uniform(0.03, 0.12) * h))
        cv2.ellipse(m, (int(rng.uniform(0, w)), int(rng.uniform(0, h))), axes,
                    float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
        m = _blur_sigma(m, 0.4 * max(axes))
        m *= rng.uniform(*s.glare_strength) / max(float(m.max()), 1e-6)
        scene += (255.0 - scene) * m[..., None]  # screen blend: bright, never above 255

    scene = _blur_sigma(scene, rng.uniform(*s.blur_sigma))
    sig = rng.uniform(*s.noise_sigma)
    if sig > 0:
        scene += rng.standard_normal(scene.shape, dtype=np.float32) * np.float32(sig)
    return np.clip(scene, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# A1.5 scene generator
# ----------------------------------------------------------------------------
def _choice(rng: np.random.Generator, probs: Dict[str, float]) -> str:
    keys = list(probs)
    p = np.asarray([float(probs[k]) for k in keys])
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


def _long_side_frac(s, kind: str, subtype: str) -> Tuple[float, float]:
    over = s.long_side_frac_by_subtype.to_dict() if hasattr(s.long_side_frac_by_subtype, "to_dict") else {}
    if subtype in over:
        return tuple(over[subtype])
    return tuple(s.doc_long_side_frac if kind == "document" else s.distractor_long_side_frac)


def _make_object(rng, s, pools: ScenePools, kind: str, subtype: str, canvas_long: int):
    """Source image for one object. Returns dict(img, alpha, text, source, template_rel_key, size_wh)."""
    tcfg = s.templates
    lo, hi = _long_side_frac(s, kind, subtype)
    target_long = canvas_long * rng.uniform(lo, hi)
    alpha, text, tpl_key = None, None, None
    if subtype == "id":
        doc = pools.midv_docs[int(rng.integers(0, len(pools.midv_docs)))]
        img, source, tpl_key = pools.midv_template(doc), f"midv500/{doc.doc_id}", f"midv/{doc.doc_id}"
    elif subtype in ("note", "receipt"):
        L = int(rng.integers(tcfg.long_side[0], tcfg.long_side[1] + 1))
        asp = rng.uniform(*(tcfg.note_aspect if subtype == "note" else tcfg.receipt_aspect))
        img, text = render_text_page(rng, (int(round(L * asp)), L), subtype, tcfg)
        source, tpl_key = "rendered", "own"
    else:
        asp = rng.uniform(*_DISTRACTOR_ASPECT[subtype])
        # distractors are rendered near their final size (no template is stored)
        L = max(16, int(round(target_long * 1.1)))
        wh = (max(8, int(round(L * asp))), L)
        tex = pools.backgrounds.sample_texture(rng, wh) if subtype in ("book", "photo") and len(pools.backgrounds) else None
        img, alpha = make_distractor_rgba(rng, subtype, wh, tcfg, texture=tex)
        source = "generated"
    hs, ws = img.shape[:2]
    sc = target_long / max(hs, ws)
    return dict(img=img, alpha=alpha, text=text, source=source, tpl_key=tpl_key, size=(ws * sc, hs * sc),
                min_scale=min(1.0, canvas_long * lo / target_long))  # never shrink below the spec range


def generate_scene(rng: np.random.Generator, cfg, pools: ScenePools, tier: str, *,
                   scene_id: str = "scene", seed: int = 0,
                   template_sink: Optional[Dict[str, np.ndarray]] = None
                   ) -> Tuple[np.ndarray, SceneAnnotation]:
    """Generate one cluttered desk scene of the given tier (PLAN A1.5).

    Returns (scene BGR uint8 (H,W,3), SceneAnnotation with paths relative to the split root).
    Template images that must be written are put in `template_sink` (rel path -> BGR).
    """
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}")
    s = cfg.synth
    cw, ch = (int(v) for v in s.canvas_wh)
    canvas_long = max(cw, ch)

    # 1. counts (T1/T2 need a pair, so at least 2 objects)
    while True:
        n_docs = int(rng.integers(s.n_docs[0], s.n_docs[1] + 1))
        n_dis = int(rng.integers(s.n_distractors[0], s.n_distractors[1] + 1))
        if tier == "T0" or n_docs + n_dis >= 2:
            break
    doc_probs = {k: float(v) for k, v in s.doc_subtype_probs.to_dict().items()}
    if not pools.midv_docs:
        doc_probs.pop("id", None)
    kinds = ["document"] * n_docs + ["distractor"] * n_dis
    subtypes = [_choice(rng, doc_probs) for _ in range(n_docs)] + \
               [_choice(rng, s.distractor_subtype_probs.to_dict()) for _ in range(n_dis)]
    # random order, but object 0 is always a document (it anchors the T1/T2 qualifying pair)
    order = [int(i) for i in rng.permutation(len(kinds))]
    first_doc = next(i for i in order if kinds[i] == "document")
    order.remove(first_doc)
    order = [first_doc] + order
    kinds, subtypes = [kinds[i] for i in order], [subtypes[i] for i in order]
    objs = [_make_object(rng, s, pools, k, st, canvas_long) for k, st in zip(kinds, subtypes)]

    # 2. one camera for the whole scene (O1); desk region that the canvas sees
    H = _camera(rng, cw, ch, float(s.perspective_jitter_frac))
    Hinv = np.linalg.inv(H)
    desk_c = _to_image(Hinv, _canvas_quad(cw, ch))
    desk_box = (float(desk_c[:, 0].min()), float(desk_c[:, 1].min()),
                float(desk_c[:, 0].max()), float(desk_c[:, 1].max()))

    # 3+4. rotations, then rejection-sampled placement. On failure objects shrink, but never below
    # the configured size range (tiny objects would fall under the detector's minimum area);
    # when that is not enough, distractors (then extra documents) are dropped.
    angles = [float(rng.uniform(-180, 180)) if rng.random() < float(s.any_rotation_prob)
              else float(rng.uniform(*s.rotation_deg_range)) for _ in objs]
    shrink = 1.0
    min_objs = 1 if tier == "T0" else 2
    quads = None
    while quads is None:
        for _ in range(int(s.max_layout_restarts)):
            f = [max(shrink, o["min_scale"]) for o in objs]
            sizes = [(o["size"][0] * fi, o["size"][1] * fi) for o, fi in zip(objs, f)]
            quads = _layout(rng, s, sizes, angles, tier, H, desk_box, (cw, ch))
            if quads is not None:
                break
            shrink *= float(s.layout_shrink)
        if quads is None:
            drop = max((i for i in range(1, len(objs)) if kinds[i] == "distractor"), default=None)
            if drop is None or len(objs) <= min_objs:
                drop = len(objs) - 1 if len(objs) > min_objs else None
            if drop is None:
                raise RuntimeError(f"{scene_id}: cannot place objects for tier {tier}")
            log.debug("%s: dropping object %d after failed layouts", scene_id, drop)
            for lst in (objs, kinds, subtypes, angles):
                del lst[drop]
            shrink = 1.0

    # background seen through the same camera: render a padded desk, then warp it
    pad_x, pad_y = int(np.ceil(desk_box[2] - desk_box[0])), int(np.ceil(desk_box[3] - desk_box[1]))
    desk_bg = pools.backgrounds.sample(rng, (pad_x, pad_y))
    T = np.array([[1, 0, desk_box[0]], [0, 1, desk_box[1]], [0, 0, 1]], np.float64)
    scene = cv2.warpPerspective(desk_bg, H @ T, (cw, ch), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT).astype(np.float32)

    # 5. z-order and compositing, back to front, with a soft drop shadow under each object
    z = [int(v) for v in rng.permutation(len(objs))]
    shadow_dir = rng.uniform(0, 2 * np.pi)  # one light: every shadow falls the same way
    ds = s.drop_shadow
    label = np.full((ch, cw), -1, np.int16)
    full_px = np.zeros(len(objs), np.int64)
    for k in sorted(range(len(objs)), key=lambda i: z[i]):
        o = objs[k]
        off = rng.uniform(*ds.offset_px)
        dx, dy = off * np.cos(shadow_dir), off * np.sin(shadow_dir)
        sig = rng.uniform(*ds.blur_sigma)
        res = _warp_object(o["img"], o["alpha"], quads[k], (cw, ch), pad=int(off + 3 * sig + 2))
        if res is None:
            continue
        x0, y0, warped, a = res
        hh, ww = a.shape
        roi = scene[y0:y0 + hh, x0:x0 + ww]
        sh = cv2.warpAffine(a, np.float32([[1, 0, dx], [0, 1, dy]]), (ww, hh))
        roi *= (1.0 - rng.uniform(*ds.opacity) * _blur_sigma(sh, sig))[..., None]
        # per-object brightness / contrast (the stored template stays un-jittered)
        warped = (warped - 128.0) * rng.uniform(*s.object_contrast) + 128.0
        warped *= rng.uniform(*s.object_gain)
        roi[:] = roi * (1.0 - a[..., None]) + np.clip(warped, 0, 255) * a[..., None]
        hard = a > 0.5
        label[y0:y0 + hh, x0:x0 + ww][hard] = k
        full_px[k] = int(hard.sum())

    # 7. visible fraction from the z-order label map (fraction NOT covered by other objects)
    vis = [float((label == k).sum()) / float(full_px[k]) if full_px[k] > 0 else 0.0 for k in range(len(objs))]

    # 6. global photometric effects
    scene = _photometric(rng, s, scene)

    objects: List[SceneObject] = []
    for k, o in enumerate(objs):
        tpl_rel = None
        if o["tpl_key"] == "own":
            tpl_rel = f"templates/{scene_id}_obj{k}.png"
        elif o["tpl_key"] is not None:
            tpl_rel = f"templates/{o['tpl_key']}.png"
        if tpl_rel is not None and template_sink is not None:
            template_sink[tpl_rel] = o["img"]
        objects.append(SceneObject(obj_id=k, kind=kinds[k], subtype=subtypes[k],
                                   quad=quads[k].astype(np.float32), z_order=z[k],
                                   visible_fraction=float(min(1.0, max(0.0, vis[k]))),
                                   source=o["source"], template_path=tpl_rel, text=o["text"]))
    ann = SceneAnnotation(scene_id=scene_id, image_path=f"scenes/{scene_id}.jpg", width=cw, height=ch,
                          tier=tier, seed=int(seed), objects=objects)
    return scene, ann


# ----------------------------------------------------------------------------
# A1.6 split builder
# ----------------------------------------------------------------------------
def scene_id_for(seed: int) -> str:
    return f"scene_{seed:06d}"


def config_hash(cfg) -> str:
    return hashlib.sha1(json.dumps(cfg.synth.to_dict(), sort_keys=True).encode("utf-8")).hexdigest()


def _build_one(seed: int, cfg, pools: ScenePools, root: Path) -> str:
    s = cfg.synth
    rng = get_rng(seed)
    tp = s.tier_probs.to_dict()
    p = np.asarray([float(tp[t]) for t in TIERS])
    tier = TIERS[int(rng.choice(len(TIERS), p=p / p.sum()))]
    sink: Dict[str, np.ndarray] = {}
    sid = scene_id_for(seed)
    scene, ann = generate_scene(rng, cfg, pools, tier, scene_id=sid, seed=seed, template_sink=sink)
    quality = int(rng.integers(s.jpeg_quality[0], s.jpeg_quality[1] + 1))
    for rel, img in sink.items():
        if not (root / rel).exists():  # shared MIDV templates are written once
            ensure_dir((root / rel).parent)
            cv2.imwrite(str(root / rel), img)
    if not cv2.imwrite(str(root / ann.image_path), scene, [cv2.IMWRITE_JPEG_QUALITY, quality]):
        raise IOError(f"cannot write {root / ann.image_path}")
    # annotation LAST: its existence marks the scene complete (resumable builds)
    write_json(root / "annotations" / f"{sid}.json", annotation_to_dict(ann))
    return sid


_WORKER: Dict[str, object] = {}


def _init_worker(cfg, split: str, raw_root, root: str) -> None:
    logging.basicConfig(level=logging.WARNING)
    _WORKER.update(cfg=cfg, pools=make_pools(cfg, split, raw_root), root=Path(root))


def _worker_build(seed: int) -> str:
    return _build_one(seed, _WORKER["cfg"], _WORKER["pools"], _WORKER["root"])


def build_split(split: str, cfg, out_root, n: Optional[int] = None, *, workers: int = 1,
                raw_root=None, progress: bool = True) -> Path:
    """Generate `n` scenes of `split` into `out_root` (resumable: finished scenes are skipped)."""
    s = cfg.synth
    n = int(n if n is not None else getattr(s.sizes, split))
    offset = int(getattr(s.seed_offsets, split))
    root = ensure_dir(out_root)
    for sub in ("scenes", "annotations", "templates/midv"):
        ensure_dir(root / sub)
    pools = make_pools(cfg, split, raw_root)
    for d in pools.midv_docs:  # shared templates written once, before any worker starts
        p = root / "templates" / "midv" / f"{d.doc_id}.png"
        if not p.exists():
            cv2.imwrite(str(p), pools.midv_template(d))
    todo = [offset + i for i in range(n)
            if not (root / "annotations" / f"{scene_id_for(offset + i)}.json").exists()]
    log.info("%s: %d of %d scenes to generate", split, len(todo), n)

    from tqdm import tqdm
    bar = tqdm(total=len(todo), desc=f"synth {split}", disable=not progress)
    if workers > 1 and len(todo) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers, initializer=_init_worker,
                                 initargs=(cfg, split, raw_root, str(root))) as ex:
            for _ in ex.map(_worker_build, todo, chunksize=4):
                bar.update(1)
    else:
        for seed in todo:
            _build_one(seed, cfg, pools, root)
            bar.update(1)
    bar.close()

    write_json(root / "manifest.json", {
        "version": DATASET_VERSION, "split": split, "n_scenes": n, "seed_offset": offset,
        "config_hash": config_hash(cfg), "contract_version": CONTRACT_VERSION,
        "midv_doc_ids": [d.doc_id for d in pools.midv_docs],
        "background_ids": list(pools.backgrounds.ids),
    })
    return root


# ----------------------------------------------------------------------------
# contact sheet (eyeball check, PLAN A1 "done when")
# ----------------------------------------------------------------------------
def contact_sheet(split_root, out_path, n: int = 24, cols: int = 6, thumb_w: int = 400) -> Path:
    """Grid of the first `n` scenes with GT quads: documents green, distractors red."""
    from .annotations import list_annotations, load_annotation
    tiles = []
    for jp in list_annotations(split_root)[:n]:
        ann = load_annotation(jp)
        img = cv2.imread(ann.image_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        for o in ann.objects:
            col = (0, 220, 0) if o.kind == "document" else (0, 0, 230)
            q = np.round(o.quad).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [q], True, col, 3, cv2.LINE_AA)
            cv2.circle(img, tuple(int(v) for v in o.quad[0]), 8, col, -1)  # own TL corner
            cv2.putText(img, f"{o.subtype} {o.visible_fraction:.2f}", tuple(int(v) for v in o.quad.mean(0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2, cv2.LINE_AA)
        cv2.putText(img, f"{ann.scene_id} {ann.tier}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                    (255, 255, 255), 3, cv2.LINE_AA)
        s = thumb_w / img.shape[1]
        tiles.append(cv2.resize(img, (thumb_w, int(img.shape[0] * s)), interpolation=cv2.INTER_AREA))
    if not tiles:
        raise FileNotFoundError(f"no scenes under {split_root}")
    th = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    tiles += [np.zeros_like(tiles[0])] * ((-len(tiles)) % cols)
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    cv2.imwrite(str(out_path), np.vstack(rows))
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build a synthetic split (PLAN A1.6)")
    ap.add_argument("--split", choices=("train", "val", "test"))
    ap.add_argument("--out", help="split root (default data/synth/<split>) or contact-sheet image")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--raw-root", default=None, help="default <paths.data_root>/raw")
    ap.add_argument("--config", default=None)
    ap.add_argument("--contact-sheet", default=None, metavar="SPLIT_ROOT",
                    help="write a contact sheet of SPLIT_ROOT to --out instead of generating")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if a.contact_sheet:
        print("wrote", contact_sheet(a.contact_sheet, a.out or "results/contact_sheet.jpg"))
    else:
        if not a.split:
            ap.error("--split is required")
        c = load_config(a.config)
        out = a.out or (resolve_path(c.paths.data_root) / "synth" / a.split)
        print("wrote", build_split(a.split, c, out, a.n, workers=a.workers, raw_root=a.raw_root))
