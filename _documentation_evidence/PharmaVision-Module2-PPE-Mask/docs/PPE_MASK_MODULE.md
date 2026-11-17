# PharmaVision AI — Module 2: PPE / Mask check (GMP-inspired awareness gate)

**What it does:** a webcam frame goes in; faces are boxed and labelled `MASK OK`, `NO MASK` or `MASK INCORRECT`; a banner and a
*gate* say whether tablet inspection may start. A face without a mask gets a red box with **NO MASK** above it; when the mask is
put on, the box turns green.

**What it is not:** a validated personnel-control system. It is a research prototype / awareness aid. A mask check does not
cover gowning, hair covers, gloves or access control, and no result here may be described as GMP-compliant.

## 1. Install (nothing is overwritten)
Extract this zip **into your project root** (`...\PharmaSight-AI-Starter-v2\PharmaSight-AI`). It only adds new folders/files:
`src\ppe_mask\`, `datasets\ppe_mask\`, `models\ppe_mask\`, `reports\ppe_mask\`, `docs\PPE_MASK_MODULE.md`, `tests\test_ppe_mask.py`,
`requirements_ppe_mask.txt`. Your existing environment (Ultralytics, OpenCV, PyTorch) is enough; this module works on your
current Python 3.14 as well as on 3.12.

```powershell
cd "C:\Users\AFZAL COMPUTERS\OneDrive\Desktop\PharmaSight-AI-Starter-v2\PharmaSight-AI"
pip install -r requirements_ppe_mask.txt
python -m pytest tests\test_ppe_mask.py
```

## 2. Dataset (the only thing you must download)
**Face Mask Detection (Kaggle, by Andrew Maranhão):** https://www.kaggle.com/datasets/andrewmvd/face-mask-detection
- 853 images, 4,072 labelled faces, PASCAL VOC boxes; classes `with_mask`, `without_mask`, `mask_weared_incorrect`.
- Licence shown on the dataset listings: CC0 1.0 (check the Kaggle page yourself before redistributing anything).
- Class counts: with_mask 3,232 · without_mask 717 · mask_weared_incorrect 123 (**very imbalanced; the "incorrect" class is too rare to be reliable**).
- Download via the Kaggle website (needs a free account), unzip, and arrange as `datasets\ppe_mask\raw\kaggle_face_mask\images` and `...\annotations`.

Optional extra data: Roboflow Universe has many "face mask" datasets exportable in *YOLOv8* format. I have not verified any specific one;
check licence, label quality and duplicates yourself. A YOLOv8 export has its own `data.yaml`, so you can pass it to
`train_mask_detector.py --data <that data.yaml>` and skip the prepare step.

## 3. Steps
```powershell
# 1) convert + split (random 70/15/15, exact duplicates removed, seed 42)
python src\ppe_mask\prepare_mask_dataset.py

# 2) quick pipeline check first (3 epochs), then the real run
python src\ppe_mask\train_mask_detector.py --epochs 3 --name mask_smoke_test
python src\ppe_mask\train_mask_detector.py

# 3) live webcam demo
python src\ppe_mask\live_mask_demo.py
```
Best weights land in `runs\ppe_mask\mask_yolov8n\weights\best.pt`. CPU training takes a while: judge the speed from the smoke test.

## 4. Your own webcam data (this is what makes it work on YOUR camera)
The public set is mostly small faces in crowd photos, very different from you sitting at a laptop. Collect your own frames:
```powershell
python src\ppe_mask\collect_webcam_faces.py --session s1_desk_daylight
```
SPACE → drag a box around the face → ENTER → press **M** (mask), **N** (no mask) or **I** (incorrect). 60–100 frames per session, with
varied distance, head angle and lighting; use a **different session name for each setup/person** (only record people who agree).
Then:
```powershell
# train on Kaggle + sessions s1 and s2; keep s3 completely unseen
python src\ppe_mask\prepare_mask_dataset.py --train-sessions s1_desk_daylight s2_evening_lamp
python src\ppe_mask\train_mask_detector.py --weights runs\ppe_mask\mask_yolov8n\weights\best.pt --name mask_yolov8n_stage2 --epochs 40
```

## 5. Evaluate honestly (run on test data ONCE, after all choices are final)
```powershell
python src\ppe_mask\evaluate_mask_detector.py --weights runs\ppe_mask\mask_yolov8n_stage2\weights\best.pt
python src\ppe_mask\evaluate_gate.py --weights runs\ppe_mask\mask_yolov8n_stage2\weights\best.pt ^
    --images datasets\ppe_mask\webcam\s3_unseen\images --labels datasets\ppe_mask\webcam\s3_unseen\labels
```
`evaluate_gate.py` reports the two errors that matter: **false accept** (an unmasked operator passes: the dangerous one) and
**false reject** (a masked operator blocked: annoying but safe). Write your acceptance limits **before** you train (for example a maximum
false-accept rate on the unseen webcam session) and put them in your report. Open every false-accept image and look at it.

## 6. How the gate decides (src\ppe_mask\gate.py — thresholds are `GateConfig`, tune on validation data, not test)
- `NOT_COMPLIANT`: any face detected as no-mask / incorrect with confidence ≥ 0.35 (deliberately low: be suspicious).
- `COMPLIANT`: every counted face is `with_mask` with confidence ≥ 0.60.
- `UNCERTAIN`: anything in between (never a silent pass). `NO_FACE`: nothing detected → inspection stays locked.
- Faces smaller than 0.2% of the frame (background people) are ignored: a limitation if you need to monitor people far away.
- The gate **opens** after 8 consecutive compliant frames and **closes** after 3 non-compliant ones.

## 7. Known limitations (say these in your video/report)
Small, single-source public dataset · rare "incorrect" class · domain gap to your webcam until you add your own frames · a mask is only
one GMP-relevant item · confidence scores are not calibrated probabilities · results on a few of your own frames are anecdotal, not validation.

## 8. Integration contract (for the merge later)
```python
from src.ppe_mask.mask_module import MaskComplianceModule
mask = MaskComplianceModule("runs/ppe_mask/mask_yolov8n_stage2/weights/best.pt")
annotated, info = mask.process(frame_bgr)       # the module never opens the camera
if info["gate_open"]:                            # also: info["status"], info["detections"]
    ...run the tablet detector/classifier...
```
