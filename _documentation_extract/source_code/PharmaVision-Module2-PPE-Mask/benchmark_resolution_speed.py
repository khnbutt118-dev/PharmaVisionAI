from pathlib import Path
import time
import statistics
import cv2

from src.ppe_mask.mask_module import MaskComplianceModule

ROOT = Path.cwd()
WEIGHTS = ROOT / "runs/ppe_mask/mask_yolov8n_full/weights/best.pt"
IMAGE_DIR = ROOT / "datasets/ppe_mask/prepared/images/val"

images = sorted(IMAGE_DIR.glob("*.png"))

if not images:
    raise RuntimeError("No validation images found.")

frames = []

for path in images:
    frame = cv2.imread(str(path))
    if frame is not None:
        frames.append(frame)

print("Images loaded:", len(frames))

for imgsz in [416, 640, 832]:

    print("\n" + "=" * 55)
    print(f"BENCHMARK: {imgsz} x {imgsz}")
    print("=" * 55)

    model = MaskComplianceModule(
        weights=str(WEIGHTS),
        conf=0.10,
        imgsz=imgsz,
        device="cpu"
    )

    # Warm-up
    for frame in frames[:5]:
        model.detect(frame)

    timings = []

    # Benchmark all validation images
    for i, frame in enumerate(frames, 1):

        start = time.perf_counter()
        model.detect(frame)
        elapsed = time.perf_counter() - start

        timings.append(elapsed)

        if i % 25 == 0 or i == len(frames):
            print(f"Processed {i}/{len(frames)}")

    timings.sort()

    average = statistics.mean(timings)
    median = statistics.median(timings)
    p95 = timings[int(0.95 * (len(timings) - 1))]
    fps = 1 / average

    print(f"\nResolution: {imgsz}")
    print(f"Average inference: {average * 1000:.2f} ms")
    print(f"Median inference: {median * 1000:.2f} ms")
    print(f"P95 inference: {p95 * 1000:.2f} ms")
    print(f"Approximate FPS: {fps:.2f}")

print("\nBenchmark completed.")
