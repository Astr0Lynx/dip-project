# docscan

Multi-document detection, separation and enhancement in cluttered scenes (DIP course project).

**Start here: read `PLAN.md`.** It is the single source of truth for who builds what.

## Setup (both members, once)
Both members use the **same Python minor version** (3.11 or 3.12; agree on one).
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                          # once it exists: pip install -r requirements-lock.txt
pip install -e .                                         # editable install is REQUIRED (config finds the repo root)
# OCR needs the Tesseract program itself (only Member B needs it):
#   Ubuntu: sudo apt install tesseract-ocr    macOS: brew install tesseract
#   Windows: install from the UB-Mannheim installer and add it to PATH
pytest -q                                                # must be green before you start
python -m docscan.data.fixtures --out data/fixtures --n 3
python -m docscan.cli run --input data/fixtures/scenes --out results/demo
python -c "import docscan; print(docscan.CONTRACT_VERSION)"   # must print the version in PLAN.md
```
Green tests do NOT prove identical environments (`>=` ranges drift; scikit-learn pickles break
across versions). In Phase 0 one member runs `pip freeze > requirements-lock.txt`, commits it,
and the other installs from it. Commit a new lock whenever a dependency changes.

## Running code
* Always run modules from the repo root as `python -m docscan.<module>`. Never run a file inside
  `src/docscan/` directly, and never start Python or Jupyter with that folder as the working
  directory: `src/docscan/types.py` then shadows the standard-library `types` module and Python
  dies at startup with a confusing `ImportError`.
* PyCharm: in each run configuration choose "Module name" (not "Script path") and set the
  working directory to the repo root.
* Relative paths in the config are relative to the repo root: open them with
  `docscan.config.resolve_path(...)`.

## Rules in one breath
1. Only edit files you own (table in PLAN.md section 3).
2. `types.py`, `geometry.py`, `validate.py`, `config.py`, `io_utils.py`, `data/annotations.py`,
   `data/fixtures.py` are FROZEN. Change them only through `CONTRACT_CHANGES.md`.
3. `pytest -q` must pass before every push. The smoke test runs the whole pipeline.
4. Log every handoff in `HANDOFF.md`.
