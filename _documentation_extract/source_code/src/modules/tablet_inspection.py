from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "models" / "pillqc_multiclass" / "best_model.pth"
)

CLASS_NAMES = ["normal", "contamination", "chipping"]
IMAGE_SIZE = 224


class TabletInspector:
    """Loads the trained PillQC classifier and predicts tablet condition."""

    def __init__(self, model_path=DEFAULT_MODEL_PATH, device=None):
        self.model_path = Path(model_path)

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Trained model not found: {self.model_path}"
            )

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=True
        )

        self.class_names = checkpoint.get("class_names", CLASS_NAMES)

        if self.class_names != CLASS_NAMES:
            raise ValueError(
                f"Unexpected model classes: {self.class_names}"
            )

        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            len(self.class_names)
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @torch.inference_mode()
    def inspect_image(self, image, review_threshold=0.70):
        """
        Predict the most likely class.

        image must be a PIL image.
        A low-confidence result is flagged for manual review.
        """

        if not isinstance(image, Image.Image):
            raise TypeError("Please provide a PIL Image.")

        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        logits = self.model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

        predicted_index = int(torch.argmax(probabilities).item())
        predicted_class = self.class_names[predicted_index]
        confidence = float(probabilities[predicted_index].item())

        probability_dict = {
            name: float(probabilities[index].item())
            for index, name in enumerate(self.class_names)
        }

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probability_dict,
            "review_required": confidence < review_threshold,
            "result_type": "MODEL_PREDICTION_ONLY"
        }


def inspect_image(image):
    """Convenience function for one-off image inspection."""
    inspector = TabletInspector()
    return inspector.inspect_image(image)
