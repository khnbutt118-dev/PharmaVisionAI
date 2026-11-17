# Changelog

## v2 (patched starter)
- requirements.txt: added `streamlit` (app.py imports it, so v1 failed after installing requirements). Moved mediapipe, pyttsx3 and ultralytics to `requirements-optional.txt`.
- New `src/data/find_duplicates.py`: exact (SHA-256) and near-duplicate (dHash) candidate pairs, cross-source flag.
- New `src/data/suggest_labels.py`: label suggestions from folder names, written to a separate file for human review.
- `qa_engine.py`: added `interpret_probability` (three-state decision from P(defective)); original `interpret_prediction` kept.
- New tests (`tests/test_data_tools.py`) and `tests/__init__.py` so `python -m pytest` finds the `src` package.
- Docs: source-specific cautions (MVTec AD train folders are normal-only, single-product sources, capping labels unverified), pre-registered gate section, updated Phase 1 checklist and README.
- Not changed: no datasets, no trained models, no empirical results are included.
