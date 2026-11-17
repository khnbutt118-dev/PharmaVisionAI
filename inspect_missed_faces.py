from pathlib import Path
from PIL import Image

root = Path("datasets/ppe_mask/prepared")

for name in ["maksssksksss113", "maksssksksss75"]:
    image_path = root / "images" / "val" / f"{name}.png"
    label_path = root / "labels" / "val" / f"{name}.txt"

    with Image.open(image_path) as image:
        width, height = image.size

    print("\n" + "=" * 55)
    print("IMAGE:", name)
    print("Dimensions:", width, "x", height)

    lines = label_path.read_text().strip().splitlines()

    for i, line in enumerate(lines, 1):
        cls, x, y, w, h = map(float, line.split())

        pixel_w = w * width
        pixel_h = h * height
        area = pixel_w * pixel_h

        classes = {
            0: "with_mask",
            1: "without_mask",
            2: "mask_weared_incorrect"
        }

        print(
            f"Face {i}: {classes[int(cls)]:25} "
            f"width={pixel_w:.1f}px "
            f"height={pixel_h:.1f}px "
            f"area={area:.1f}px"
        )
