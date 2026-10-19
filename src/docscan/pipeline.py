"""
docscan.pipeline -- end-to-end glue. Owner: B (already working with the stubs).

    process_scene(scene_path, cfg, out_dir, detections_override=None, *,
                  save_images=True, on_doc=None) -> PipelineResult

Per scene:  (downscale to cfg.pipeline.max_scene_side) -> detect_all -> (resolve_overlaps on
ALL detections) -> for each ACCEPTED document, SEQUENTIALLY:
rectify -> correct_illumination -> binarize_all -> write files / on_doc -> free memory.

Output layout (out_dir/<scene_id>/), <id> = det_id, 2 digits:
    detections.json            FINAL detections (after the overlap stage), ALL incl. rejected
    detections_raw.json        cascade output before the overlap stage (only if overlap.enabled)
    overlay.jpg                scene with the final quads drawn          (save_images only)
    doc_<id>_rectified.png     BGR rectified crop                        (save_images only)
    doc_<id>_gray.png          illumination-corrected gray               (save_images only)
    doc_<id>_bin_<method>.png  one per method in cfg.binarize.methods    (save_images only)
    doc_<id>_meta.json         quad, size, timings
    result.json                the PipelineResult
"""
from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .detect.api import classify_candidates, detect_all
from .enhance.binarize import binarize_all
from .enhance.illumination import correct_illumination
from .enhance.rectify import rectify
from .geometry import scale_quad
from .io_utils import ensure_dir, read_image, write_image, write_json
from .overlap.resolve import resolve_overlaps
from .types import (BinarizedDoc, Detection, DocRecord, EnhancedDoc, PipelineResult,
                    RectifiedDoc, detection_to_dict)
from .validate import validate_detections
from .viz import draw_detections

# on_doc(det, rect, enh, bins) runs for every accepted document while its arrays are still
# alive, so the benchmark can score in memory instead of writing and re-reading PNGs.
DocCallback = Callable[[Detection, RectifiedDoc, EnhancedDoc, Dict[str, BinarizedDoc]], None]


def _limit_scene_size(scene: np.ndarray, max_side: int) -> Tuple[np.ndarray, float, float]:
    """Downscale so the long side is <= max_side (PLAN section 2 memory rule).
    Returns (scene, sx, sy) with processed = original * (sx, sy)."""
    h, w = scene.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s >= 1.0:
        return scene, 1.0, 1.0
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    small = cv2.resize(scene, (nw, nh), interpolation=cv2.INTER_AREA)  # AREA avoids aliasing
    return small, nw / w, nh / h


def _scale_detection(d: Detection, sx: float, sy: float) -> Detection:
    contour = None
    if d.contour is not None:
        contour = np.round(d.contour * np.array([sx, sy])).astype(np.int32)
    return dataclasses.replace(d, quad=scale_quad(d.quad, sx, sy), contour=contour)


def process_scene(scene_path, cfg, out_dir, detections_override=None, *,
                  save_images: bool = True, on_doc: Optional[DocCallback] = None) -> PipelineResult:
    """Run the whole pipeline on one scene image and write its outputs.

    detections_override: optional List[Detection] in ORIGINAL scene coordinates that replaces
        detect_all (use data.annotations.oracle_detections to test the back half without
        Member A's detector).
    save_images: False skips overlay.jpg and the per-document PNGs (benchmark runs).
    on_doc: optional DocCallback; its run time is reported as timings_s["on_doc"] and is
        NOT included in timings_s["total"].
    """
    t0 = time.perf_counter()
    scene_path = Path(scene_path)
    scene_id = scene_path.stem
    out = ensure_dir(Path(out_dir) / scene_id)
    scene = read_image(scene_path)
    h0, w0 = scene.shape[:2]
    scene, sx, sy = _limit_scene_size(scene, getattr(cfg.get("pipeline"), "max_scene_side", 2000))
    timings: Dict[str, float] = {}

    t = time.perf_counter()
    if detections_override is not None:
        dets_raw = [_scale_detection(d, sx, sy) for d in detections_override]
    else:
        dets_raw = detect_all(scene, cfg)
    timings["detect"] = time.perf_counter() - t

    dets = dets_raw
    if cfg.overlap.enabled:
        t = time.perf_counter()
        # ALL detections in, ALL out (v1.1): the stage needs the rejected ones to keep det_ids
        # unique and to split merged blobs that the cascade rejected. Split proposals are
        # judged by A's public classify_candidates, injected here so overlap/ never imports
        # detect/. Oracle runs have no detector to defer to, hence classify=None.
        classify = None if detections_override is not None else classify_candidates
        dets = resolve_overlaps(scene, dets_raw, cfg, classify=classify)
        timings["overlap"] = time.perf_counter() - t
        write_json(out / "detections_raw.json", [detection_to_dict(d) for d in dets_raw])
    if cfg.debug_validate:
        validate_detections(dets, scene.shape[:2])

    write_json(out / "detections.json", [detection_to_dict(d) for d in dets])
    if save_images:
        write_image(out / "overlay.jpg", draw_detections(scene, dets))
    accepted = [d for d in dets if d.accepted]

    records: List[DocRecord] = []
    cb_time = 0.0
    for det in accepted:
        tt: Dict[str, float] = {}
        t = time.perf_counter()
        rect = rectify(scene, det, cfg)
        tt["rectify"] = time.perf_counter() - t

        t = time.perf_counter()
        enh = correct_illumination(rect, cfg)
        tt["illum"] = time.perf_counter() - t

        t = time.perf_counter()
        bins = binarize_all(enh, cfg)
        tt["binarize"] = time.perf_counter() - t

        # named by det_id (not list position) so files join directly with detections.json
        prefix = f"doc_{det.det_id:02d}"
        files: Dict[str, str] = {}
        if save_images:
            files = {"rectified": f"{prefix}_rectified.png", "gray": f"{prefix}_gray.png"}
            write_image(out / files["rectified"], rect.image)
            write_image(out / files["gray"], enh.gray)
            for m, b in bins.items():
                files[f"bin_{m}"] = f"{prefix}_bin_{m}.png"
                write_image(out / files[f"bin_{m}"], b.binary)
        if on_doc is not None:
            t = time.perf_counter()
            on_doc(det, rect, enh, bins)
            cb_time += time.perf_counter() - t

        h, w = rect.image.shape[:2]
        rec = DocRecord(det_id=det.det_id, quad=det.quad, out_size=(w, h), files=files, timings_s=tt)
        records.append(rec)
        write_json(out / f"{prefix}_meta.json",
                   {"det_id": det.det_id, "quad": det.quad.tolist(), "out_size": [w, h],
                    "files": files, "timings_s": tt, "illum_method": enh.illum_method})
        del rect, enh, bins  # keep peak memory ~ one document

    if on_doc is not None:
        timings["on_doc"] = cb_time
    timings["total"] = time.perf_counter() - t0 - cb_time
    result = PipelineResult(scene_id=scene_id, scene_path=str(scene_path), out_dir=str(out),
                            n_candidates=len(dets_raw), n_accepted=len(accepted),
                            detections_file="detections.json", docs=records, timings_s=timings)
    write_json(out / "result.json", {
        "scene_id": result.scene_id, "scene_path": result.scene_path,
        "n_candidates": result.n_candidates, "n_accepted": result.n_accepted,
        # every coordinate in this folder lives in the processed frame = original * scale
        "scene_wh_original": [w0, h0], "scene_scale": [sx, sy],
        "timings_s": result.timings_s, "contract_version": result.contract_version,
        "docs": [{"det_id": r.det_id, "quad": r.quad.tolist(), "out_size": list(r.out_size),
                  "files": r.files, "timings_s": r.timings_s} for r in records]})
    return result
