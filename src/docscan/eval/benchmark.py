"""
Member B owns this file. STUB.  Runs everything on a split and writes the report.

    python -m docscan.eval.benchmark --split test --run-id r1 [--config path.yaml]
        [--split-root PATH]      # any split root instead of data/synth/<split> (fixtures, midv_real)
        [--oracle]               # GT detections: isolates B's stages
        [--stages all|detect]    # detect = detection metrics only (cascade / overlap ablations)
        [--limit N] [--workers N] [--gallery 12]
    python -m docscan.eval.benchmark --dibco --run-id d1     # illumination x binarization grid

Writes results/<run_id>/metrics.json and results/<run_id>/report.md (PLAN B6).
Per-document metrics are computed in memory through process_scene(..., on_doc=...);
images are saved only for the gallery scenes (save_images=False elsewhere).
"""
from __future__ import annotations


def main(argv=None):
    raise NotImplementedError("Member B: implement (PLAN B6)")


if __name__ == "__main__":
    main()
