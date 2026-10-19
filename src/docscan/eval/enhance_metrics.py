"""
Member B owns this file. STUB.

PUBLIC CONTRACT (B-internal, listed so A knows they exist):
    fmeasure(pred_bin, gt_bin) -> float          # ink (0) is the positive class
    psnr(pred_bin, gt_bin)     -> float
    upright_rotation_k(rect_quad, gt_quad) -> int          # v1.1: np.rot90(img, k) makes the crop upright
    rectification_ssim(rect_bgr, template_bgr, k=None) -> float  # k given: that rotation only; else best of 4
    aspect_ratio_err(rect_wh, template_wh, k) -> float     # v1.1: |log(aspect_out / aspect_tpl)| after rotation
    residual_skew_from_gt(H, gt_quad) -> float             # v1.1: exact, degrees, uses the GT quad
    residual_skew_deg(gray)    -> float          # GT-free projection-profile estimate, in [-10, 10]
"""
from __future__ import annotations


def fmeasure(pred_bin, gt_bin):
    raise NotImplementedError("Member B: implement (PLAN B5)")


def psnr(pred_bin, gt_bin):
    raise NotImplementedError("Member B: implement (PLAN B5)")


def upright_rotation_k(rect_quad, gt_quad):
    raise NotImplementedError("Member B: implement (PLAN B5)")


def rectification_ssim(rect_bgr, template_bgr, k=None):
    raise NotImplementedError("Member B: implement (PLAN B5)")


def aspect_ratio_err(rect_wh, template_wh, k):
    raise NotImplementedError("Member B: implement (PLAN B5)")


def residual_skew_from_gt(H, gt_quad):
    raise NotImplementedError("Member B: implement (PLAN B5)")


def residual_skew_deg(gray):
    raise NotImplementedError("Member B: implement (PLAN B5)")
