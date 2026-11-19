<p align="center"><img src="https://raw.githubusercontent.com/khnbutt118-dev/khnbutt118-dev/main/assets/medical-ai-banner.svg" alt="Hassan Butt — Pharmacy meets Artificial Intelligence" width="100%"></p>

<p align="center"><a href="https://github.com/khnbutt118-dev">← Back to GitHub profile</a> · <a href="https://hassanbuttportfolio.vercel.app/">Portfolio</a> · <a href="https://www.linkedin.com/in/hassan-butt-ab0036334/">LinkedIn</a></p>

---

# PharmaSight AI
**Real-Time Pharmaceutical Quality & GMP Intelligence System**  
Research and decision-support prototype — not a GMP-validated inspection system.

## What this repository provides
A phased, modular foundation for:
1. Tablet appearance inspection (first modelling target)
2. Blister-pack completeness inspection
3. Visible PPE monitoring
4. Gesture-based interaction
5. Voice alerts and QA-oriented inspection records

The modules are separate specialist pipelines connected by one application. Do not train one model to perform all tasks.

## Start here
1. Extract this ZIP.
2. Open the `PharmaSight-AI` folder in VS Code.
3. Follow [`docs/PHASE_1_CHECKLIST.md`](docs/PHASE_1_CHECKLIST.md).
4. Put downloaded data in the matching `datasets/raw/...` folder. Keep each source in its own subfolder.
5. Run the audit scripts before training anything.

### Windows PowerShell setup
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
Optional packages for later phases (gestures, voice, YOLO) are in `requirements-optional.txt`; install them only when you reach those phases.

If activation is blocked in the current terminal:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```
For PyTorch, select the appropriate CPU/CUDA install command from the official PyTorch installation selector if the default install is unsuitable for your machine.

### First checks
```powershell
python -c "import torch, cv2; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('OpenCV:', cv2.__version__)"
python webcam_test.py
```
Launch the placeholder dashboard with `streamlit run app.py`. Press `Q` to close the webcam window. Webcam access may require closing other camera applications and granting Windows camera permission.

## Dataset placement
```text
datasets/
├── raw/
│   ├── tablet_defects/
│   │   ├── source_name_1/
│   │   └── source_name_2/
│   ├── blister_packs/
│   │   └── source_name/
│   └── ppe/
│       └── source_name/
├── processed/                 # generated outputs; do not edit manually
└── dataset_manifest.csv       # generated/maintained inventory
```
Keep original downloads unchanged in `raw`. Unzip each dataset into its own named source folder. Do not mix classes from different sources until labels, licensing, and annotation compatibility have been audited.

## Phase 1 audit commands
```powershell
python -m src.data.validate_images --root datasets/raw/tablet_defects --recursive
python -m src.data.build_manifest --root datasets/raw/tablet_defects --output datasets/dataset_manifest.csv --task tablet
python -m src.data.find_duplicates --root datasets/raw/tablet_defects
python -m src.data.suggest_labels
```
The manifest is an inventory, not a verified ground-truth label file. Review `label`, `defect_type`, `licence`, and `split` fields before training. The manifest builder deliberately leaves labels empty. `suggest_labels` writes *suggestions* derived from folder names to a separate file (`datasets/label_suggestions.csv`) to speed up manual review; it never edits the manifest and its output must be checked against each dataset's documentation.
`find_duplicates` lists exact and near-duplicate candidate pairs (flagging pairs that cross source folders) in `reports/dataset_audit/duplicate_candidates.csv`; review them visually before changing anything.

Run the tests with `python -m pytest`.

## Current status
- Project scaffold and Phase 1 utilities: included.
- Real datasets: **not included**; obtain them from their original sources and check licence/terms.
- Trained model weights: **not included**.
- Empirical model performance: **not available until data is supplied and experiments are run**.

## Research safeguards
- Use source-, batch-, physical-sample-, or capture-session-separated splits where possible. Avoid near-duplicate leakage.
- Hold out at least one source/session for external testing.
- Report sensitivity/recall for defective samples, false-pass rate, precision, specificity, per-class F1, confusion matrix, and latency.
- Use an `uncertain / manual review` state. A low-confidence prediction is not a pass.
- Image predictions do not establish manufacturing root cause.
- A prototype does not replace qualified human inspection, AQL sampling, SOPs, or batch-release decisions.

See `docs/` for the data protocol, validation plan, project scope, and phased roadmap.
