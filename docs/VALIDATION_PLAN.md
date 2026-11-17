# Model validation plan

For the tablet task, define positive class as `defective` before calculating metrics.

Report:
- confusion matrix
- defective-class precision and recall/sensitivity
- specificity for normal samples
- false-pass rate = defective samples predicted normal / all truly defective samples
- per-class F1 and macro-F1
- performance by dataset/source and capture condition
- inference latency (state hardware and image size)
- uncertainty/manual-review rate and performance at the chosen threshold

Do not report accuracy alone. Tune thresholds on validation data, not the external test set. Evaluate the final model once on the held-out test source/session. Inspect false negatives individually. Grad-CAM is an explanation aid, not proof of correct reasoning.

A model confidence score is not a calibrated probability unless calibration has been evaluated. The prototype is not a GMP-validated system and cannot make batch-release decisions.

## Pre-registered gate (fill in BEFORE training)
Write your acceptance criteria here and commit them before running any experiment, for example: minimum defective recall and maximum false-pass rate on the held-out source, and the maximum acceptable manual-review rate. Choose the numbers yourself with a QA supervisor; do not copy placeholder values.

## Decision thresholds
`src/modules/qa_engine.py::interpret_probability` maps P(defective) to attention / uncertain / no-visible-defect using placeholder thresholds (0.10 and 0.90). Tune them on validation data only, so that the pre-registered gate is met. Only a very low P(defective) is reported as "no visible defect"; everything between the thresholds goes to manual review.
