"""
docscan.types  --  FROZEN CONTRACT v1.1   (v1.0 -> v1.1: see CONTRACT_CHANGES.md / PLAN_CHANGES.md)

This file is the single source of truth for every object that crosses the
boundary between Member A (detection) and Member B (rectify/enhance/eval).
DO NOT edit without a Contract Change Request (see CONTRACT_CHANGES.md).

GLOBAL CONVENTIONS (apply everywhere in the project)
----------------------------------------------------
* Colour images : np.ndarray, dtype uint8, shape (H, W, 3), BGR order (OpenCV).
* Gray images   : np.ndarray, dtype uint8, shape (H, W).
* Binary images : np.ndarray, dtype uint8, values ONLY {0, 255}.
                  0 = INK (foreground), 255 = PAPER (background).
                  (This matches the DIBCO ground-truth convention.)
* Coordinates   : (x, y) in pixels, origin = top-left, y grows downward.
                  Unless stated otherwise, coordinates live in the FULL-RESOLUTION
                  scene image (never in a downscaled working copy).
* Quad          : np.ndarray, dtype float32, shape (4, 2), clockwise on screen.
                  - Predictions (Detection.quad) are ordered by
                    geometry.order_quad(): index 0 = top-left-most point.
                  - Ground truth (SceneObject.quad) is ordered in the DOCUMENT'S
                    OWN frame: TL, TR, BR, BL of the printed page, whatever its
                    rotation in the scene. Therefore ALWAYS compare prediction to
                    ground truth modulo a cyclic shift (geometry.match_quads and
                    geometry.best_cyclic_alignment already do this).
* Floats in [0,1] are probabilities / fractions. No NaN / inf anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

CONTRACT_VERSION = "1.1"

Quad = np.ndarray  # alias, see conventions above

# Allowed string values (validated in docscan.validate)
SOURCES = ("contour", "hough", "overlap_resolved")
DECIDERS = ("heuristic", "ml", "dl")
BINARIZE_METHODS = ("otsu", "sauvola", "niblack", "tuned")
ILLUM_METHODS = ("none", "bgsub", "homomorphic")
OBJECT_KINDS = ("document", "distractor")
DOC_SUBTYPES = ("id", "receipt", "note")
DISTRACTOR_SUBTYPES = ("blank_paper", "folder", "book", "photo", "device")
TIERS = ("T0", "T1", "T2")  # T0: no overlap, T1: overlap < 20%, T2: overlap 20-50%


# ----------------------------------------------------------------------------
# Detection side  (produced by Member A)
# ----------------------------------------------------------------------------
@dataclass
class Candidate:
    """A quadrilateral region that MIGHT be a document (before classification)."""
    cand_id: int
    quad: Quad                       # (4,2) float32, ordered, full-res coords
    contour: Optional[np.ndarray]    # (N,1,2) int32 full-res OpenCV contour, or None
    source: str                      # "contour" | "hough" | "overlap_resolved" (split proposals, v1.1)
    area: float                      # polygon area of quad, px^2, full-res
    quad_fit: float = 1.0            # contour_area / quad_area, clipped to [0,1].
                                     # ~1.0 = clean rectangle; low = ragged / merged / occluded


@dataclass
class Detection:
    """A candidate after the rejection cascade. `detect_all` returns ALL of them."""
    det_id: int                      # unique within a scene, >= 0
    quad: Quad                       # (4,2) float32, ordered, full-res coords
    score: float                     # P(real document) in [0,1] from the deciding stage
    accepted: bool                   # True = real document, False = rejected distractor
    decided_by: str                  # "heuristic" | "ml" | "dl"
    source: str                      # "contour" | "hough" | "overlap_resolved"
    contour: Optional[np.ndarray] = None   # (N,1,2) int32 full-res, may be None
    quad_fit: float = 1.0            # same meaning as Candidate.quad_fit
    occlusion_est: float = 0.0       # 0..1 estimated hidden fraction (set by overlap stage)
    features: Optional[np.ndarray] = None  # (F,) float32 feature vector (debug / training only)


# ----------------------------------------------------------------------------
# Enhancement side  (produced by Member B)
# ----------------------------------------------------------------------------
@dataclass
class RectifiedDoc:
    det_id: int
    image: np.ndarray                # BGR uint8 (h, w, 3), fronto-parallel crop
    H: np.ndarray                    # (3,3) float64, maps SCENE coords -> RECTIFIED coords
    quad: Quad                       # ordered source quad in scene coords used for the warp


@dataclass
class EnhancedDoc:
    det_id: int
    gray: np.ndarray                 # uint8 (h, w) illumination-corrected gray
    illum_method: str                # one of ILLUM_METHODS
    shadow_map: Optional[np.ndarray] = None  # uint8 (h, w) estimated background (viz only)


@dataclass
class BinarizedDoc:
    det_id: int
    binary: np.ndarray               # uint8 (h, w) in {0,255}; 0 = ink
    method: str                      # one of BINARIZE_METHODS


# ----------------------------------------------------------------------------
# Pipeline output  (paths only; NO image arrays, so memory stays flat)
# ----------------------------------------------------------------------------
@dataclass
class DocRecord:
    det_id: int
    quad: Quad
    out_size: Tuple[int, int]        # (width, height) of rectified image
    files: Dict[str, str]            # logical name -> path RELATIVE to the scene output dir
                                     # names: "rectified", "gray", "bin_otsu", "bin_sauvola",
                                     #        "bin_niblack", "bin_tuned"
    timings_s: Dict[str, float] = field(default_factory=dict)


@dataclass
class PipelineResult:
    scene_id: str
    scene_path: str
    out_dir: str                     # results/<run_id>/<scene_id>
    n_candidates: int
    n_accepted: int
    detections_file: str             # "detections.json" (relative to out_dir)
    docs: List[DocRecord]
    timings_s: Dict[str, float] = field(default_factory=dict)
    contract_version: str = CONTRACT_VERSION


# ----------------------------------------------------------------------------
# Ground truth / dataset side  (written by Member A's generator, read by both)
# ----------------------------------------------------------------------------
@dataclass
class SceneObject:
    obj_id: int
    kind: str                        # "document" | "distractor"
    subtype: str                     # DOC_SUBTYPES or DISTRACTOR_SUBTYPES
    quad: Quad                       # (4,2) float32, DOCUMENT'S OWN frame order (TL,TR,BR,BL)
    z_order: int                     # higher = closer to camera
    visible_fraction: float          # 0..1 fraction of the object not covered by others
    source: str = ""                 # e.g. "midv500/01_alb_id", "rendered", "sroie/X001"
    template_path: Optional[str] = None  # clean upright image of the document (documents only)
    text: Optional[str] = None       # exact rendered text (note/receipt documents only), for OCR eval


@dataclass
class SceneAnnotation:
    scene_id: str
    image_path: str                  # absolute path after load_annotation()
    width: int
    height: int
    tier: str                        # one of TIERS
    seed: int
    objects: List[SceneObject]


# ----------------------------------------------------------------------------
# (De)serialisation helpers -- both agents MUST use these, never hand-roll JSON.
# ----------------------------------------------------------------------------
def _quad_to_list(q: np.ndarray) -> List[List[float]]:
    return [[float(x), float(y)] for x, y in np.asarray(q, dtype=np.float32).reshape(4, 2)]


def detection_to_dict(d: Detection, include_contour: bool = False,
                      include_features: bool = False) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "det_id": int(d.det_id),
        "quad": _quad_to_list(d.quad),
        "score": float(d.score),
        "accepted": bool(d.accepted),
        "decided_by": d.decided_by,
        "source": d.source,
        "quad_fit": float(d.quad_fit),
        "occlusion_est": float(d.occlusion_est),
    }
    if include_contour and d.contour is not None:
        out["contour"] = d.contour.reshape(-1, 2).astype(int).tolist()
    if include_features and d.features is not None:
        out["features"] = [float(v) for v in d.features]
    return out


def detection_from_dict(o: Dict[str, Any]) -> Detection:
    contour = None
    if "contour" in o:
        contour = np.asarray(o["contour"], dtype=np.int32).reshape(-1, 1, 2)
    feats = np.asarray(o["features"], dtype=np.float32) if "features" in o else None
    return Detection(
        det_id=int(o["det_id"]),
        quad=np.asarray(o["quad"], dtype=np.float32).reshape(4, 2),
        score=float(o["score"]),
        accepted=bool(o["accepted"]),
        decided_by=o["decided_by"],
        source=o["source"],
        contour=contour,
        quad_fit=float(o.get("quad_fit", 1.0)),
        occlusion_est=float(o.get("occlusion_est", 0.0)),
        features=feats,
    )


def annotation_to_dict(a: SceneAnnotation) -> Dict[str, Any]:
    """Paths stored in JSON are RELATIVE to the split root (the folder that
    contains `scenes/`, `annotations/`, `templates/`)."""
    return {
        "scene_id": a.scene_id,
        "image": a.image_path,
        "width": int(a.width),
        "height": int(a.height),
        "tier": a.tier,
        "seed": int(a.seed),
        "objects": [
            {
                "obj_id": int(o.obj_id),
                "kind": o.kind,
                "subtype": o.subtype,
                "quad": _quad_to_list(o.quad),
                "z_order": int(o.z_order),
                "visible_fraction": float(o.visible_fraction),
                "source": o.source,
                "template": o.template_path,
                "text": o.text,
            }
            for o in a.objects
        ],
    }


def annotation_from_dict(d: Dict[str, Any]) -> SceneAnnotation:
    objs = [
        SceneObject(
            obj_id=int(o["obj_id"]),
            kind=o["kind"],
            subtype=o["subtype"],
            quad=np.asarray(o["quad"], dtype=np.float32).reshape(4, 2),
            z_order=int(o["z_order"]),
            visible_fraction=float(o["visible_fraction"]),
            source=o.get("source", ""),
            template_path=o.get("template"),
            text=o.get("text"),
        )
        for o in d["objects"]
    ]
    return SceneAnnotation(
        scene_id=d["scene_id"], image_path=d["image"], width=int(d["width"]),
        height=int(d["height"]), tier=d["tier"], seed=int(d["seed"]), objects=objs,
    )
