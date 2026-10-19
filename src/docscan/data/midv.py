"""
docscan.data.midv -- Member A. Loader for the raw MIDV-500 (and MIDV-2019) download (PLAN A1.1).

PUBLIC (used only inside A's generator):
    index_midv(raw_root)                -> List[MidvDoc]     # one entry per document type
    rectified_template(midv_doc, rng)   -> np.ndarray (BGR)  # clean upright document image
    split_midv(docs, fracs, seed)       -> Dict[split, List[MidvDoc]]

Expected layout (one folder per document type, e.g. after unzipping `01_alb_id.zip`):

    <raw_root>/.../01_alb_id/
        images/01_alb_id.tif            template image of the clean document
        images/<CLIP>/<CLIP>01_01.tif   video frames, CLIP in TS TA KS KA HS HA PS PA CS CA
        ground_truth/01_alb_id.json     {"quad": [[x,y] x4]} of the document in the template
        ground_truth/<CLIP>/<CLIP>01_01.json

VERIFIED on the real download (2026-10-14, 4 doc types; `pip install --no-deps midv500` fetches the
zips into <raw_root>/midv500/): templates ship as `images/<doc_id>.tif` and ARE the clean document
(no border), and the template json holds per-field boxes ("field01", "photo", ...), not a document
quad, so the whole template image is used. Clips TS TA KS KA HS HA PS PA CS CA (T=table,
K=keyboard, H=hand, P=partial, C=clutter; S/A = Samsung / Apple), 30 `.tif` frames each, 1080x1920.
Frame GT is {"quad": [[x,y] x4]} in the document's OWN TL,TR,BR,BL order (it follows the card's
rotation). Each zip also carries a `videos/` folder that nothing here uses. Some GT json files are
malformed (the midv500 package mentions 29_irn_drvlic): `read_quad` returns None for those.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..geometry import quad_edge_lengths
from ..io_utils import read_json

log = logging.getLogger("docscan.data.midv")

_DOC_DIR = re.compile(r"^(\d{2})_([a-z]{3})_(\w+)$")
_FRAME_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
_TEMPLATE_LONG_SIDE = (800, 1000)  # PLAN A1.1


@dataclass
class MidvDoc:
    """One MIDV document type (e.g. `01_alb_id`)."""
    doc_id: str
    country: str
    doc_type: str                               # "id", "passport", "drvlic", ...
    root: Path
    template_image: Optional[Path] = None
    template_gt: Optional[Path] = None
    clips: Dict[str, List[Tuple[Path, Path]]] = field(default_factory=dict)  # clip -> [(frame, gt)]


def _find_image(folder: Path, stem: str) -> Optional[Path]:
    for ext in _FRAME_EXTS:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def index_midv(raw_root) -> List[MidvDoc]:
    """Find every MIDV document folder under `raw_root` (sorted by doc_id; [] if none)."""
    raw_root = Path(raw_root)
    if not raw_root.exists():
        log.warning("MIDV root %s does not exist", raw_root)
        return []
    docs: Dict[str, MidvDoc] = {}
    for gt_dir in sorted(raw_root.rglob("ground_truth")):
        root = gt_dir.parent
        m = _DOC_DIR.match(root.name)
        if not gt_dir.is_dir() or not m or root.name in docs:
            continue
        img_dir = root / "images"
        doc = MidvDoc(doc_id=root.name, country=m.group(2), doc_type=m.group(3), root=root,
                      template_image=_find_image(img_dir, root.name),
                      template_gt=(gt_dir / f"{root.name}.json") if (gt_dir / f"{root.name}.json").exists() else None)
        for clip_gt in sorted(p for p in gt_dir.iterdir() if p.is_dir()):
            pairs = []
            for gj in sorted(clip_gt.glob("*.json")):
                fr = _find_image(img_dir / clip_gt.name, gj.stem)
                if fr is not None:
                    pairs.append((fr, gj))
            if pairs:
                doc.clips[clip_gt.name] = pairs
        docs[doc.doc_id] = doc
    if not docs:
        log.warning("no MIDV document folders found under %s", raw_root)
    return [docs[k] for k in sorted(docs)]


def split_midv(docs: Sequence[MidvDoc], fracs: Sequence[float] = (0.70, 0.15, 0.15),
               seed: int = 42) -> Dict[str, List[MidvDoc]]:
    """Fixed split by document TYPE (no leakage of a document across splits): shuffle the
    sorted doc ids with `seed`, then cut 70/15/15."""
    ids = sorted(d.doc_id for d in docs)
    perm = np.random.default_rng(seed).permutation(len(ids))
    n_tr = int(round(fracs[0] * len(ids)))
    n_va = int(round(fracs[1] * len(ids)))
    by_id = {d.doc_id: d for d in docs}
    order = [ids[int(i)] for i in perm]
    parts = {"train": order[:n_tr], "val": order[n_tr:n_tr + n_va], "test": order[n_tr + n_va:]}
    return {k: [by_id[i] for i in sorted(v)] for k, v in parts.items()}


def read_quad(gt_json: Path) -> Optional[np.ndarray]:
    """GT quad (4,2) float32 from a MIDV json, or None if missing / malformed."""
    try:
        q = np.asarray(read_json(gt_json)["quad"], dtype=np.float32).reshape(4, 2)
    except (KeyError, ValueError, OSError, TypeError):
        return None
    return q if np.all(np.isfinite(q)) else None


def _warp_quad(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Warp the quad (document order TL,TR,BR,BL) to an upright rectangle."""
    e = quad_edge_lengths(quad)
    w, h = int(round(max(e[0], e[2]))), int(round(max(e[1], e[3])))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _fit_long_side(img: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Clamp the long side into [lo, hi] (deterministic: templates are stored once per doc)."""
    long_side = max(img.shape[:2])
    target = min(max(long_side, lo), hi)
    if target == long_side:
        return img
    s = target / long_side
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, (max(1, int(round(img.shape[1] * s))), max(1, int(round(img.shape[0] * s)))),
                      interpolation=interp)


def _frame_fallback(doc: MidvDoc) -> Optional[np.ndarray]:
    """No template: warp the largest fully visible document from a table clip (clean condition)."""
    best, best_area = None, 0.0
    clips = [c for c in sorted(doc.clips) if c.startswith("T")] or sorted(doc.clips)
    for clip in clips:
        for frame_path, gt in doc.clips[clip]:
            q = read_quad(gt)
            if q is None:
                continue
            img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            if not np.all((q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)):
                continue
            area = float(cv2.contourArea(q))
            if area > best_area:
                best, best_area = _warp_quad(img, q), area
    return best


def rectified_template(midv_doc: MidvDoc, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Clean upright document image (BGR), long side clamped into [800, 1000] px.

    Deterministic for a given document (the generator stores each template ONCE and it must
    be the exact image that was composited); `rng` is accepted for the frozen signature.
    Raises FileNotFoundError when neither a template nor a usable frame exists.
    """
    img = None
    if midv_doc.template_image is not None:
        img = cv2.imread(str(midv_doc.template_image), cv2.IMREAD_COLOR)
        q = read_quad(midv_doc.template_gt) if midv_doc.template_gt is not None else None
        if img is not None and q is not None:
            img = _warp_quad(img, q)
    if img is None:
        img = _frame_fallback(midv_doc)
    if img is None:
        raise FileNotFoundError(f"MIDV {midv_doc.doc_id}: no template image and no usable frame")
    return _fit_long_side(img, *_TEMPLATE_LONG_SIDE)


def export_midv_real(raw_root, out_root, every_nth: int = 3) -> None:
    """A1.7 (after the MVP): real clutter frames of the test-split document types."""
    raise NotImplementedError("Member A: A1.7 is scheduled after the MVP (needs the MIDV layout verified)")


def download_midv500(dst, attempts: int = 3) -> List[str]:
    """Resumable MIDV-500 download (needs `pip install --no-deps midv500`: its opencv-python
    dependency would clash with opencv-python-headless). Uses the package's link list and
    download/unzip helpers; skips document folders that are already extracted, retries each
    file `attempts` times, and returns the doc ids still missing (stops at the first file
    the server keeps refusing: Smart Engines' FTP blocked us for days after ~3 GB)."""
    import time
    from midv500.download_dataset import midv500_links
    from midv500.utils import download, unzip

    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    names = [link.split("/")[-1][:-4] for link in midv500_links]
    for link, name in zip(midv500_links, names):
        if (dst / name / "ground_truth").is_dir():
            continue
        for attempt in range(1, attempts + 1):
            zp = dst / f"{name}.zip"
            try:
                if zp.exists():
                    zp.unlink()  # a partial zip from an interrupted run
                download(link, str(dst))
                unzip(str(zp), str(dst))
                zp.unlink()
                log.info("MIDV %s ok", name)
                break
            except Exception as e:  # network errors come in many types (URLError, socket, ftplib)
                log.warning("MIDV %s attempt %d failed: %r", name, attempt, e)
                time.sleep(60 * attempt)
        else:
            break
    return [n for n in names if not (dst / n / "ground_truth").is_dir()]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="MIDV-500 helpers")
    ap.add_argument("--download", metavar="DST", help="resumable download into DST (e.g. data/raw/midv500)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if a.download:
        missing = download_midv500(a.download)
        print("all 50 documents present" if not missing else f"{len(missing)} still missing, first: {missing[0]}")
    else:
        ap.print_help()
