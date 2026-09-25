from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import CNNBaseline


# -----------------------------
# Paths
# -----------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "best_cnn_baseline.pth"


# -----------------------------
# Device
# -----------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# -----------------------------
# Load model
# -----------------------------

model = CNNBaseline()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)
model.eval()


# -----------------------------
# Image transformation
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


# -----------------------------
# Prediction function
# -----------------------------

def predict_image(image_path):

    image_path = Path(image_path)

    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[0, predicted_class].item()

    class_names = ["Cat", "Dog"]

    print(f"Prediction: {class_names[predicted_class]}")
    print(f"Confidence: {confidence:.2%}")