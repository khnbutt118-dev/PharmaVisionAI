# Phase 1 — Foundation and dataset audit

## Milestone A: local environment
- [ ] Extract ZIP and open project root in VS Code.
- [ ] Create/activate `.venv`.
- [ ] Install dependencies from `requirements.txt`.
- [ ] Confirm PyTorch, OpenCV and webcam availability.
- [ ] Run `python webcam_test.py`.

## Milestone B: dataset research
For each candidate dataset, record:
- [ ] Exact source URL, paper/repository and access date.
- [ ] Licence/terms and permitted research, commercial use, and redistribution.
- [ ] Number of images and exact labels.
- [ ] Image format, resolution, colour channels and acquisition conditions.
- [ ] Annotation type (image label, bounding box, mask, cavity labels).
- [ ] Known duplicate/near-duplicate risks and physical sample identity.
- [ ] Whether the classes genuinely map to our taxonomy.

Candidate leads from the project brief: PillQC/pillQC, MVTec AD pill category, PharmaVision-related releases, and blister/PPE datasets. These are leads only; verify the exact release, download availability, licence and labels before use.

## Milestone C: inventory and data protocol
- [ ] Download and extract each source into its own `datasets/raw/<task>/<source_name>/` folder.
- [ ] Run image validation and create the manifest.
- [ ] Run the duplicate finder; review cross-source pairs before defining splits.
- [ ] Run `suggest_labels`, verify suggestions against each dataset's documentation, then copy confirmed labels into the manifest.
- [ ] Manually fill source, dataset, label, defect type, licence, split and sample/session metadata.
- [ ] Define leakage-safe train/validation/test partitions.
- [ ] Hold out an entire source or capture session for external testing when feasible.
- [ ] Record camera distance, lighting, orientation, background and occlusion for original captures.
- [ ] Document limitations and class imbalance in `reports/dataset_audit/`.

## Commands
```powershell
python -m src.data.validate_images --root datasets/raw/tablet_defects --recursive
python -m src.data.build_manifest --root datasets/raw/tablet_defects --output datasets/dataset_manifest.csv --task tablet
python -m src.data.find_duplicates --root datasets/raw/tablet_defects
python -m src.data.suggest_labels
python -m src.data.prepare_data --manifest datasets/dataset_manifest.csv
```

Do not proceed to Phase 2 until labels and licences are reviewed and the split strategy is documented.
