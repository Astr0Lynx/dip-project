"""
docscan.data.assets -- Member A. Source images for the synthetic scene generator (PLAN A1.2-A1.4).

PUBLIC (used by data/synth.py only):
    render_text_page(rng, size_wh, subtype)  -> (BGR, text)   # subtype "note" | "receipt"
    make_distractor(rng, subtype, size_wh)   -> BGR           # DISTRACTOR_SUBTYPES
    BackgroundPool(raw_root, split)          .sample(rng, canvas_wh) -> BGR
                                             .sample_texture(rng, size_wh) -> BGR

Everything works with NO downloaded data: backgrounds and textures fall back to procedural
noise when `data/raw/dtd` (and the optional DocCornerDataset negatives) are missing.
"""
from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger("docscan.data.assets")

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_SPLITS = ("train", "val", "test")


# ----------------------------------------------------------------------------
# fonts and words
# ----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _font_dir() -> Path:
    import matplotlib  # DejaVu fonts ship with matplotlib, so no system font is needed
    return Path(matplotlib.get_data_path()) / "fonts" / "ttf"


@lru_cache(maxsize=64)
def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(_font_dir() / name), size)


@lru_cache(maxsize=1)
def load_wordlist() -> Tuple[str, ...]:
    """Real English words (OCR is more reliable on real words than on random strings)."""
    words = (Path(__file__).with_name("wordlist.txt")).read_text(encoding="utf-8").split()
    return tuple(sorted(set(words)))


def _words(rng: np.random.Generator, n: int) -> List[str]:
    wl = load_wordlist()
    return [wl[int(i)] for i in rng.integers(0, len(wl), n)]


# ----------------------------------------------------------------------------
# procedural noise (fallback textures, paper grain, desks)
# ----------------------------------------------------------------------------
def _value_noise(rng: np.random.Generator, h: int, w: int, cell: float) -> np.ndarray:
    """Smooth noise in [0,1]: a coarse random grid upsampled with cubic interpolation."""
    gh, gw = max(2, int(h / cell) + 2), max(2, int(w / cell) + 2)
    g = rng.random((gh, gw)).astype(np.float32)
    return np.clip(cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC), 0.0, 1.0)


def _fractal(rng: np.random.Generator, h: int, w: int,
             cells: Sequence[float] = (0.5, 0.15, 0.04)) -> np.ndarray:
    """Sum of value-noise octaves (cell sizes as fractions of the long side), in [0,1]."""
    long_side = max(h, w)
    acc = np.zeros((h, w), np.float32)
    wsum = 0.0
    for i, c in enumerate(cells):
        wt = 0.5 ** i
        acc += wt * _value_noise(rng, h, w, max(2.0, c * long_side))
        wsum += wt
    acc /= wsum
    lo, hi = float(acc.min()), float(acc.max())
    return (acc - lo) / max(hi - lo, 1e-6)


def _colorize(field: np.ndarray, c0: Sequence[float], c1: Sequence[float]) -> np.ndarray:
    """Map a [0,1] field to a BGR gradient between two colours."""
    c0, c1 = np.asarray(c0, np.float32), np.asarray(c1, np.float32)
    img = c0[None, None, :] + field[..., None] * (c1 - c0)[None, None, :]
    return np.clip(img, 0, 255).astype(np.uint8)


def _rand_color(rng: np.random.Generator, lo: int = 30, hi: int = 220) -> np.ndarray:
    return rng.integers(lo, hi + 1, 3).astype(np.float32)


def procedural_texture(rng: np.random.Generator, size_wh: Tuple[int, int]) -> np.ndarray:
    """Colourful low-frequency texture (book covers, photo prints) when no DTD files exist."""
    w, h = size_wh
    f = _fractal(rng, h, w, cells=(rng.uniform(0.2, 0.7), rng.uniform(0.05, 0.2), 0.02))
    return _colorize(f, _rand_color(rng), _rand_color(rng))


def procedural_desk(rng: np.random.Generator, size_wh: Tuple[int, int]) -> np.ndarray:
    """Desk / table background: wood grain, fabric-like noise, or a plain surface."""
    w, h = size_wh
    style = rng.choice(["wood", "fabric", "plain"], p=[0.4, 0.3, 0.3])
    if style == "wood":
        # grain: stretched noise along the board direction, warped by a sine
        base = _value_noise(rng, h, w, cell=max(h, w) * 0.4)
        y = np.arange(h, dtype=np.float32)[:, None] / max(h, 1)
        grain = 0.5 + 0.5 * np.sin(2 * np.pi * (rng.uniform(6, 20) * y + 1.5 * base))
        f = 0.3 * grain + 0.7 * _fractal(rng, h, w)  # mostly irregular: strong stripes look fake
        c0 = np.array([rng.uniform(20, 60), rng.uniform(50, 100), rng.uniform(90, 160)], np.float32)
        img = _colorize(f, c0, c0 * rng.uniform(1.2, 1.6))
        if rng.random() < 0.5:
            img = np.ascontiguousarray(np.rot90(cv2.resize(img, (h, w))))
    elif style == "fabric":
        f = 0.7 * _fractal(rng, h, w) + 0.3 * rng.random((h, w)).astype(np.float32)
        c = _rand_color(rng, 30, 200)
        img = _colorize(f, c * 0.75, np.minimum(c * 1.2, 255))
    else:
        # includes bright desks: white paper on a white desk is a realistic hard case
        c = _rand_color(rng, 40, 235)
        img = _colorize(_fractal(rng, h, w, cells=(0.8, 0.3)), c * 0.9, np.minimum(c * 1.08, 255))
    return img


def _paper_grain(rng: np.random.Generator, img: np.ndarray, sigma: float) -> np.ndarray:
    """Fine noise plus slow brightness variation, like real paper under a camera."""
    h, w = img.shape[:2]
    slow = (_value_noise(rng, h, w, max(h, w) * 0.5) - 0.5) * 10.0
    fine = rng.normal(0.0, sigma, (h, w)).astype(np.float32)
    return np.clip(img.astype(np.float32) + (slow + fine)[..., None], 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# A1.2 rendered text pages (documents with known text)
# ----------------------------------------------------------------------------
def _paper_rgb(rng: np.random.Generator, subtype: str) -> Tuple[int, int, int]:
    if subtype == "receipt":  # thermal paper: slightly grey / cool
        v = int(rng.integers(228, 250))
        return (v, v, min(255, v + int(rng.integers(0, 6))))
    v = int(rng.integers(238, 254))  # white to cream
    return (v, v - int(rng.integers(0, 8)), v - int(rng.integers(0, 22)))


def _ink_rgb(rng: np.random.Generator, subtype: str) -> Tuple[int, int, int]:
    if subtype == "note" and rng.random() < 0.3:  # blue ballpoint-like ink
        return (int(rng.integers(10, 50)), int(rng.integers(20, 60)), int(rng.integers(90, 160)))
    v = int(rng.integers(10, 70)) if subtype == "note" else int(rng.integers(30, 90))  # receipts fade
    return (v, v, v)


def _render_note(rng, draw: ImageDraw.ImageDraw, w: int, h: int, ink, tcfg) -> List[str]:
    family = str(rng.choice(["DejaVuSans", "DejaVuSerif"]))
    size = int(round(h * rng.uniform(*tcfg.note_font_frac)))
    font = _font(f"{family}.ttf", size)
    bold = _font(f"{family}-Bold.ttf", int(size * 1.25))
    margin_x, margin_y = int(w * rng.uniform(0.07, 0.11)), int(h * rng.uniform(0.06, 0.10))
    usable_w = w - 2 * margin_x
    line_h = int(size * rng.uniform(1.35, 1.7))
    n_lines = int(rng.integers(tcfg.note_lines[0], tcfg.note_lines[1] + 1))

    lines: List[str] = []
    y = margin_y
    if rng.random() < 0.6:  # a title line
        title = " ".join(_words(rng, int(rng.integers(1, 4)))).title()
        if draw.textlength(title, font=bold) <= usable_w and y + int(size * 1.25) <= h - margin_y:
            draw.text((margin_x, y), title, font=bold, fill=ink)
            lines.append(title)
            y += int(line_h * 1.5)
    while len(lines) < n_lines and y + size <= h - margin_y:
        # fill each line to a random fraction of the width (ragged right, like real notes)
        target = usable_w * rng.uniform(0.55, 1.0)
        words: List[str] = []
        for wd in _words(rng, 20):
            trial = " ".join(words + [wd])
            if draw.textlength(trial, font=font) > target:
                break
            words.append(wd)
        if not words:
            break
        line = " ".join(words)
        if rng.random() < 0.15:
            line = line.capitalize() + "."
            if draw.textlength(line, font=font) > usable_w:
                line = " ".join(words)
        draw.text((margin_x, y), line, font=font, fill=ink)
        lines.append(line)
        y += line_h
    return lines


def _render_receipt(rng, draw: ImageDraw.ImageDraw, w: int, h: int, ink, tcfg) -> List[str]:
    n_chars = int(rng.integers(tcfg.receipt_chars[0], tcfg.receipt_chars[1] + 1))
    margin_x = int(w * 0.06)
    # DejaVu Sans Mono advance width is 0.602 em: choose the size so n_chars fill the width
    size = max(6, int((w - 2 * margin_x) / (n_chars * 0.602)))
    font = _font("DejaVuSansMono.ttf", size)
    line_h = int(size * rng.uniform(1.25, 1.45))
    y, y_max = int(h * 0.04), h - int(h * 0.03) - size
    lines: List[str] = []

    def put(text: str, align: str = "left") -> bool:
        nonlocal y
        if y > y_max:
            return False
        text = text[:n_chars]
        pad = n_chars - len(text)
        if align == "center":
            text = " " * (pad // 2) + text
        draw.text((margin_x, y), text, font=font, fill=ink)
        lines.append(text.rstrip())
        y += line_h
        return True

    def priced(name: str, price: float) -> str:
        p = f"{price:.2f}"
        return name[: n_chars - len(p) - 1].ljust(n_chars - len(p)) + p

    put(" ".join(_words(rng, int(rng.integers(1, 3)))).upper(), "center")
    put(f"{int(rng.integers(1, 999))} {_words(rng, 1)[0].upper()} STREET", "center")
    put(f"{int(rng.integers(1, 29)):02d}/{int(rng.integers(1, 13)):02d}/20{int(rng.integers(18, 27))}"
        f" {int(rng.integers(0, 24)):02d}:{int(rng.integers(0, 60)):02d}")
    put("-" * n_chars)
    total, n_items = 0.0, int(rng.integers(3, 25))
    for _ in range(n_items):
        if y > y_max - 4 * line_h:  # keep room for the total block
            break
        price = float(rng.integers(50, 5000)) / 100.0
        name = " ".join(_words(rng, int(rng.integers(1, 3)))).upper()
        if put(priced(name, price)):
            total += price
    put("-" * n_chars)
    tax = round(total * 0.08, 2)
    put(priced("TAX", tax))
    put(priced("TOTAL", total + tax))
    put("THANK YOU", "center")
    return lines


def render_text_page(rng: np.random.Generator, size_wh: Tuple[int, int], subtype: str,
                     tcfg=None) -> Tuple[np.ndarray, str]:
    """Render a `note` (serif/sans prose) or `receipt` (monospace items + prices) page.

    Returns (BGR uint8 (h,w,3), text). `text` is EXACTLY the visible lines joined by "\\n":
    lines that would not fit are never drawn and never included (OCR ground truth).
    `tcfg` is `cfg.synth.templates` (defaults from the YAML when None).
    """
    if subtype not in ("note", "receipt"):
        raise ValueError(f"render_text_page: subtype must be note|receipt, got {subtype!r}")
    if tcfg is None:
        from ..config import load_config
        tcfg = load_config().synth.templates
    w, h = int(size_wh[0]), int(size_wh[1])
    pil = Image.new("RGB", (w, h), _paper_rgb(rng, subtype))
    draw = ImageDraw.Draw(pil)
    ink = _ink_rgb(rng, subtype)
    lines = (_render_note if subtype == "note" else _render_receipt)(rng, draw, w, h, ink, tcfg)
    bgr = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    return _paper_grain(rng, bgr, sigma=float(rng.uniform(1.0, 3.0))), "\n".join(lines)


# ----------------------------------------------------------------------------
# A1.3 distractors
# ----------------------------------------------------------------------------
def _blank_paper(rng, w: int, h: int, tcfg, texture=None) -> np.ndarray:
    v = int(rng.integers(225, 254))
    img = np.full((h, w, 3), (v - int(rng.integers(0, 20)), v - int(rng.integers(0, 8)), v), np.uint8)
    if rng.random() < tcfg.ruled_prob:
        # ruled paper WITHOUT text: periodic lines look like text rows, a deliberate hard negative
        gap = max(4, int(h * rng.uniform(0.025, 0.04)))
        blue = (int(rng.integers(180, 230)), int(rng.integers(150, 200)), int(rng.integers(100, 160)))
        thick = max(1, h // 600)
        for y in range(int(h * rng.uniform(0.08, 0.15)), h - gap // 2, gap):
            cv2.line(img, (0, y), (w - 1, y), blue, thick, cv2.LINE_AA)
        if rng.random() < 0.7:
            x = int(w * rng.uniform(0.1, 0.15))
            cv2.line(img, (x, 0), (x, h - 1), (90, 90, 220), thick, cv2.LINE_AA)
    return _paper_grain(rng, img, sigma=float(rng.uniform(1.0, 3.0)))


def _folder(rng, w: int, h: int, tcfg, texture=None) -> np.ndarray:
    palette = [(80, 170, 215), (160, 90, 40), (40, 40, 170), (60, 140, 60), (90, 90, 90), (40, 160, 200)]
    c = np.asarray(palette[int(rng.integers(0, len(palette)))], np.float32) * rng.uniform(0.8, 1.2)
    if rng.random() < 0.5:
        img = _colorize(np.linspace(0, 1, h, dtype=np.float32)[:, None].repeat(w, 1), c, c * rng.uniform(0.7, 0.95))
    else:
        img = np.clip(np.full((h, w, 3), c), 0, 255).astype(np.uint8)
    # a darker tab band along the top and a crease near it
    tab_h = max(2, int(h * rng.uniform(0.05, 0.1)))
    x0 = int(w * rng.uniform(0.0, 0.5))
    x1 = min(w, x0 + int(w * rng.uniform(0.25, 0.45)))
    img[:tab_h, x0:x1] = np.clip(img[:tab_h, x0:x1].astype(np.float32) * 0.8, 0, 255).astype(np.uint8)
    cv2.line(img, (0, tab_h), (w - 1, tab_h), tuple(float(v) * 0.7 for v in c), max(1, h // 300), cv2.LINE_AA)
    if rng.random() < 0.4:  # a white label sticker
        lw, lh = int(w * rng.uniform(0.2, 0.35)), int(h * rng.uniform(0.06, 0.1))
        lx, ly = int(w * rng.uniform(0.1, 0.6)), int(h * rng.uniform(0.2, 0.6))
        cv2.rectangle(img, (lx, ly), (lx + lw, ly + lh), (235, 240, 240), -1)
    return _paper_grain(rng, img, sigma=float(rng.uniform(1.0, 2.5)))


def _book(rng, w: int, h: int, tcfg, texture=None) -> np.ndarray:
    img = texture.copy() if texture is not None else procedural_texture(rng, (w, h))
    # darken / tint so covers are not paper-like
    img = np.clip(img.astype(np.float32) * rng.uniform(0.5, 0.95), 0, 255).astype(np.uint8)
    border = max(2, int(min(w, h) * rng.uniform(0.03, 0.06)))
    col = tuple(int(v) for v in _rand_color(rng, 120, 240))
    cv2.rectangle(img, (border, border), (w - 1 - border, h - 1 - border), col, max(1, border // 3), cv2.LINE_AA)
    spine = int(w * rng.uniform(0.05, 0.1))
    img[:, :spine] = np.clip(img[:, :spine].astype(np.float32) * 0.6, 0, 255).astype(np.uint8)
    if rng.random() < tcfg.book_title_prob:
        # large title text: text-like, but not a document (hard negative)
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        title = " ".join(_words(rng, int(rng.integers(1, 3)))).upper()
        size = max(8, int(h * rng.uniform(0.06, 0.1)))
        font = _font("DejaVuSerif-Bold.ttf", size)
        while size > 8 and draw.textlength(title, font=font) > 0.8 * (w - spine):
            size = int(size * 0.85)
            font = _font("DejaVuSerif-Bold.ttf", size)
        x = spine + int((w - spine - draw.textlength(title, font=font)) / 2)
        draw.text((x, int(h * rng.uniform(0.15, 0.35))), title, font=font, fill=col[::-1])
        img = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    return img


def _photo(rng, w: int, h: int, tcfg, texture=None) -> np.ndarray:
    border = max(2, int(min(w, h) * rng.uniform(0.04, 0.08)))
    v = int(rng.integers(230, 252))
    img = np.full((h, w, 3), v, np.uint8)
    iw, ih = w - 2 * border, h - 2 * border - (int(h * 0.12) if rng.random() < 0.3 else 0)  # polaroid chin
    ih = max(2, ih)
    inner = cv2.resize(texture, (iw, ih)) if texture is not None else procedural_texture(rng, (iw, ih))
    img[border:border + ih, border:border + iw] = inner
    return img


def _device(rng, w: int, h: int, tcfg, texture=None) -> Tuple[np.ndarray, np.ndarray]:
    body = int(rng.integers(10, 50))
    img = np.full((h, w, 3), body, np.uint8)
    bez = max(2, int(min(w, h) * rng.uniform(0.04, 0.08)))
    screen = np.array([rng.uniform(15, 60), rng.uniform(15, 60), rng.uniform(20, 80)], np.float32)
    img[bez:h - bez, bez:w - bez] = screen.astype(np.uint8)
    # diagonal glare gradient over the glass
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = (xx / max(w, 1) + yy / max(h, 1)) / 2.0
    c = rng.uniform(0.2, 0.6)
    glare = np.exp(-((t - c) ** 2) / (2 * rng.uniform(0.03, 0.1) ** 2)) * rng.uniform(40, 110)
    img = np.clip(img.astype(np.float32) + glare[..., None], 0, 255).astype(np.uint8)
    # rounded corners: alpha mask so the desk shows through the corners
    alpha = np.zeros((h, w), np.uint8)
    r = max(1, int(min(w, h) * rng.uniform(0.04, 0.1)))
    cv2.rectangle(alpha, (r, 0), (w - 1 - r, h - 1), 255, -1)
    cv2.rectangle(alpha, (0, r), (w - 1, h - 1 - r), 255, -1)
    for cx, cy in ((r, r), (w - 1 - r, r), (w - 1 - r, h - 1 - r), (r, h - 1 - r)):
        cv2.circle(alpha, (cx, cy), r, 255, -1, cv2.LINE_AA)
    return img, alpha


_DISTRACTORS = {"blank_paper": _blank_paper, "folder": _folder, "book": _book,
                "photo": _photo, "device": _device}


def make_distractor_rgba(rng: np.random.Generator, subtype: str, size_wh: Tuple[int, int],
                         tcfg=None, texture: Optional[np.ndarray] = None
                         ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Like `make_distractor` but also returns an alpha mask (uint8, 255 = opaque) or None.
    `texture` (BGR, size_wh) is used for book covers and photo prints when given."""
    if subtype not in _DISTRACTORS:
        raise ValueError(f"unknown distractor subtype {subtype!r}")
    if tcfg is None:
        from ..config import load_config
        tcfg = load_config().synth.templates
    w, h = max(2, int(size_wh[0])), max(2, int(size_wh[1]))
    if texture is not None:
        texture = cv2.resize(texture, (w, h), interpolation=cv2.INTER_AREA)
    out = _DISTRACTORS[subtype](rng, w, h, tcfg, texture)
    return out if isinstance(out, tuple) else (out, None)


def make_distractor(rng: np.random.Generator, subtype: str, size_wh: Tuple[int, int]) -> np.ndarray:
    """Distractor image (BGR uint8, size_wh) of one of DISTRACTOR_SUBTYPES."""
    return make_distractor_rgba(rng, subtype, size_wh)[0]


# ----------------------------------------------------------------------------
# A1.4 backgrounds
# ----------------------------------------------------------------------------
def hash_split(key: str, fracs: Sequence[float] = (0.70, 0.15, 0.15), seed: int = 42) -> str:
    """Stable split for a source identity (file path / id): independent of listing order."""
    hv = int(hashlib.sha1(f"{seed}:{key}".encode("utf-8")).hexdigest()[:12], 16) / float(16 ** 12)
    edges = np.cumsum(fracs)
    for name, e in zip(_SPLITS, edges):
        if hv < e:
            return name
    return _SPLITS[-1]


class BackgroundPool:
    """Background (desk) images of ONE split, from DTD and optional DocCornerDataset negatives.

    Files are split 70/15/15 by a hash of their path relative to `raw_root`, so a background
    file never appears in two splits. Falls back to procedural desks when no file is found.
    """

    SOURCES = ("dtd", "doccorner_neg")

    def __init__(self, raw_root, split: str, fracs: Sequence[float] = (0.70, 0.15, 0.15),
                 seed: int = 42, categories: Optional[Sequence[str]] = None,
                 procedural_prob: float = 0.0):
        """`categories`: DTD classes allowed as DESK backgrounds (None = all); textures for book
        covers / photos use every file of the split. `procedural_prob`: share of procedural desks."""
        if split not in _SPLITS:
            raise ValueError(f"split must be one of {_SPLITS}")
        self.raw_root = Path(raw_root)
        self.split = split
        files: List[Path] = []
        for src in self.SOURCES:
            d = self.raw_root / src
            if d.is_dir():
                files += [p for p in d.rglob("*") if p.suffix.lower() in _IMG_EXTS]
        rel = sorted(p.relative_to(self.raw_root).as_posix() for p in files)
        self.ids: List[str] = [r for r in rel if hash_split(r, fracs, seed) == split]
        allowed = set(categories) if categories is not None else None
        # DTD paths are dtd/images/<category>/<file>; other sources are always desk-like
        self.desk_ids: List[str] = [r for r in self.ids if allowed is None or not r.startswith("dtd/")
                                    or (len(r.split("/")) >= 4 and r.split("/")[2] in allowed)]
        self.procedural_prob = float(procedural_prob)
        if not self.ids:
            log.warning("no background images under %s for split %s: using procedural desks",
                        self.raw_root, split)

    def __len__(self) -> int:
        return len(self.ids)

    def _read(self, rng: np.random.Generator, ids: Optional[List[str]] = None) -> Optional[np.ndarray]:
        ids = self.ids if ids is None else ids
        for _ in range(5):  # skip unreadable files instead of crashing a long build
            rid = ids[int(rng.integers(0, len(ids)))]
            img = cv2.imread(str(self.raw_root / rid), cv2.IMREAD_COLOR)
            if img is not None and min(img.shape[:2]) >= 16:
                return img
            log.warning("unreadable background %s", rid)
        return None

    def sample(self, rng: np.random.Generator, canvas_wh: Tuple[int, int]) -> np.ndarray:
        """Random crop with the canvas aspect, random flip, resized to canvas_wh (BGR)."""
        w, h = int(canvas_wh[0]), int(canvas_wh[1])
        use_file = bool(self.desk_ids) and rng.random() >= self.procedural_prob
        img = self._read(rng, self.desk_ids) if use_file else None
        if img is None:
            return procedural_desk(rng, (w, h))
        ih, iw = img.shape[:2]
        # largest crop with the canvas aspect, scaled by U(0.7,1): little upsampling of small files
        s = rng.uniform(0.7, 1.0)
        cw = int(min(iw, ih * w / h) * s)
        ch = int(cw * h / w)
        x0 = int(rng.integers(0, iw - cw + 1))
        y0 = int(rng.integers(0, ih - ch + 1))
        crop = img[y0:y0 + ch, x0:x0 + cw]
        if rng.random() < 0.5:
            crop = crop[:, ::-1]
        if rng.random() < 0.5:
            crop = crop[::-1, :]
        return cv2.resize(np.ascontiguousarray(crop), (w, h), interpolation=cv2.INTER_LINEAR)

    def sample_texture(self, rng: np.random.Generator, size_wh: Tuple[int, int]) -> np.ndarray:
        """Texture patch for book covers / photo prints (same split, so no leakage)."""
        img = self._read(rng) if self.ids else None
        if img is None:
            return procedural_texture(rng, size_wh)
        return cv2.resize(img, (int(size_wh[0]), int(size_wh[1])), interpolation=cv2.INTER_AREA)
