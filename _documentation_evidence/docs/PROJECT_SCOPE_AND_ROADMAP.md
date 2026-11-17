# Product scope and phased roadmap

## Architecture
One application, several specialist pipelines:
- Tablet defects: transfer-learning classifier first; localization only if suitable annotations exist.
- Blister completeness: controlled template/image-registration baseline, then detection/segmentation if needed.
- PPE: visible mask/PPE detector; presence detection is not full GMP compliance.
- Gestures: MediaPipe Hands and explicit interaction rules.
- Voice/QA: TTS plus curated, rule-based knowledge; possible causes are investigation prompts, not confirmed causes.
- Integration: camera/task router, dashboard, traceable inspection records and reports.

## Phases
1. Foundation and dataset audit.
2. Tablet normal/defective classifier.
3. Defect localization and explainability where annotations support it.
4. Blister cavity completeness.
5. PPE monitoring.
6. Gesture and voice interaction.
7. QA interpretation and structured reports.
8. Integration, external validation and demonstration.

## Intended use and limitations
Research/decision-support prototype only. Not a certified or GMP-validated inspection system, not a replacement for SOPs, qualified human QA review, AQL sampling, or batch-release decisions.
