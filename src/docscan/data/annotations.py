"""
docscan.data.annotations -- FROZEN shared dataset readers (contract v1.0).

Dataset split root layout (written by Member A's generator, read by everyone):

    <split_root>/
        scenes/<scene_id>.jpg
        annotations/<scene_id>.json      (schema = types.annotation_to_dict)
        templates/<scene_id>_obj<k>.png  (clean upright document images)
        manifest.json                    ({"version":..., "n_scenes":..., "seed_offset":..., "config_hash":...})

Paths inside the JSON are relative to <split_root>.
(v1.1) A template may be SHARED by many scenes (e.g. templates/midv/<doc_id>.png) so
MIDV templates are stored once, not once per scene. Whatever the path, the template must
be the exact source image that was composited, BEFORE any photometric jitter.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from ..geometry import order_quad
from ..io_utils import read_json
from ..types import Detection, SceneAnnotation, annotation_from_dict


def list_annotations(split_root) -> List[Path]:
    return sorted((Path(split_root) / "annotations").glob("*.json"))


def load_annotation(json_path) -> SceneAnnotation:
    """Load one annotation and resolve image / template paths to absolute paths."""
    json_path = Path(json_path)
    root = json_path.parent.parent
    ann = annotation_from_dict(read_json(json_path))
    ann.image_path = str((root / ann.image_path).resolve())
    for o in ann.objects:
        if o.template_path:
            o.template_path = str((root / o.template_path).resolve())
    return ann


def oracle_detections(ann: SceneAnnotation, min_visible: float = 0.0) -> List[Detection]:
    """Ground-truth documents as perfect Detections (accepted=True). Lets Member B
    develop rectification / enhancement / evaluation WITHOUT Member A's detector."""
    out: List[Detection] = []
    for o in ann.objects:
        if o.kind != "document" or o.visible_fraction < min_visible:
            continue
        out.append(Detection(det_id=len(out), quad=order_quad(o.quad), score=1.0, accepted=True,
                             decided_by="heuristic", source="contour", quad_fit=1.0))
    return out
