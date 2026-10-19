"""
docscan.cli -- command line entry points. Owner: B.

    python -m docscan.cli run --input <image or folder> --out results/demo [--config path.yaml]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .pipeline import process_scene

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="docscan")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="process one image or a folder of images")
    r.add_argument("--input", required=True)
    r.add_argument("--out", default="results/demo")
    r.add_argument("--config", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    p = Path(args.input)
    files = sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXT) if p.is_dir() else [p]
    for f in files:
        res = process_scene(f, cfg, args.out)
        print(f"{res.scene_id}: {res.n_accepted}/{res.n_candidates} accepted, "
              f"{res.timings_s['total']:.2f}s")


if __name__ == "__main__":
    main()
