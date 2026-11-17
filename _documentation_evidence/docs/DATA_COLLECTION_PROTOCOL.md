# Data collection and curation protocol

## Initial task
Start with tablet binary classification: `normal` versus `defective`. Retain defect subtype only when the source annotation supports it. Do not infer a subtype from an unverified filename.

## Folder convention
`datasets/raw/tablet_defects/<dataset_or_capture_source>/`

Keep original source folders intact. Never overwrite originals with resized, augmented or relabelled images. Store derived data under `datasets/processed/` and document the transformation.

## Image capture notes
For any original images, record:
- physical sample ID (pseudonymous code, not product/batch confidential details)
- capture session/date
- camera/device and approximate distance
- lighting setup and background
- tablet orientation and visible surfaces
- focus, blur, occlusion and image quality
- label and who verified it

Use a consistent, safe, non-destructive setup. Do not photograph identifiable people or confidential manufacturing records without permission.

## Labels
Initial binary labels:
- `normal`: no visible defect in the inspected view
- `defective`: visible abnormality is present

Possible subtype labels (only when supported): `chipping`, `cracking`, `capping`, `contamination`. An image may not show enough evidence to distinguish these; use an unresolved/unknown annotation rather than guessing.

## Data quality
Check readability, duplicates, class counts, resolution, blur, lighting, class definitions and label provenance. Maintain source-level metadata and licence evidence.

## Splitting
Group by source, physical tablet/sample, batch where appropriate, or capture session. Do not split near-identical images of the same sample across train and test. Reserve an unseen source/session for external testing when possible.

## Source-specific cautions (verify against each dataset's own documentation)
- **MVTec AD (pill/capsule):** the official `train` folders contain only normal images; defective images are in `test`. Do not build a supervised normal-vs-defective set from `train` alone, and keep the official test images together when you assign splits.
- **Single-product or fixed-setup sources (for example pillQC):** images may come from one tablet product under one imaging setup. Treat the whole source as one group when splitting, and prefer it as an external test source rather than mixing it into training.
- **Defect subtypes:** do not assume any source contains labelled capping images. Count the subtype labels each source actually provides before promising a multi-class model.
- **Folder-name labels:** `suggest_labels` proposes labels from folder names only. A folder name is not ground truth.
