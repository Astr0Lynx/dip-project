"""
Member B owns this file. STUB.  OCR is a SECONDARY check only.

    run_ocr(img_gray_or_bin, cfg, k=None) -> (text, mean_conf)
        # pytesseract. k given (benchmark: orientation known from GT, see upright_rotation_k):
        # OCR once after np.rot90(img, k). k None (real photos): try cfg.eval.ocr.rotations and
        # keep the one with the highest mean word confidence.
    cer(ref, hyp)                 -> float               # character error rate (rapidfuzz Levenshtein)
"""
from __future__ import annotations


def run_ocr(img, cfg, k=None):
    raise NotImplementedError("Member B: implement (PLAN B5)")


def cer(ref, hyp):
    raise NotImplementedError("Member B: implement (PLAN B5)")
