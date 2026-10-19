"""
Member B owns this file (v1.1: moved from A to B, see PLAN B0). STUB.
Detection + rejection-cascade metrics. Member A imports it (read-only) for tuning/training.

PUBLIC CONTRACT:
    evaluate_scene(dets_all, ann, cfg)  -> dict       # per-scene raw counts (see PLAN B0)
    aggregate(per_scene: List[dict])    -> dict       # final metrics, keys fixed in PLAN B0
    candidate_labels(quads, ann, cfg, duplicates="ignore") -> List[int]
        # 1 / 0 / -1 (ignored) per quad. ONE definition shared by the rejection metrics
        # (duplicates="ignore") and A's train_cascade (duplicates="positive"), so training
        # labels and evaluation labels can never drift apart.
"""
from __future__ import annotations


def evaluate_scene(dets_all, ann, cfg):
    raise NotImplementedError("Member B: implement (PLAN B0)")


def aggregate(per_scene):
    raise NotImplementedError("Member B: implement (PLAN B0)")


def candidate_labels(quads, ann, cfg, duplicates="ignore"):
    raise NotImplementedError("Member B: implement (PLAN B0)")
