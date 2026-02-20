"""
AIMonk Multilabel Classification — Inference Script
====================================================
Usage: python inference.py --image path/to/image.jpg --model best_model.pth
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── MODEL ARCHITECTURE ──────────────────────────────────────
class ProductionResNet34(nn.Module):
    def __init__(self, num_classes=4, dropout=0.4):
        super().__init__()
        self.backbone = models.resnet34(weights=None)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# ── PREPROCESSING ───────────────────────────────────────────
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

ATTR_NAMES = ['Attribute 1', 'Attribute 2', 'Attribute 3', 'Attribute 4']

# ── PREDICT ─────────────────────────────────────────────────
def predict(image_path, model_path, threshold=0.5, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = ProductionResNet34(num_classes=4, dropout=0.4).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load and preprocess image
    img = np.array(Image.open(image_path).convert('RGB'))
    transformed = transform(image=img)['image'].unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        logits = model(transformed)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    binary = (probs > threshold).astype(int)

    # Print results
    print(f"\n{'='*50}")
    print(f"  IMAGE: {image_path}")
    print(f"  THRESHOLD: {threshold}")
    print(f"{'─'*50}")

    present = []
    for i, (name, prob, pred) in enumerate(zip(ATTR_NAMES, probs, binary)):
        status = "PRESENT" if pred == 1 else "ABSENT"
        symbol = "✓" if pred == 1 else "✗"
        print(f"  {symbol} {name}: {status}  (confidence: {prob*100:.1f}%)")
        if pred == 1:
            present.append(name)

    print(f"{'─'*50}")
    if present:
        print(f"  Attributes present: {present}")
    else:
        print(f"  No attributes detected.")
    print(f"{'='*50}")

    return {'probabilities': probs, 'binary': binary, 'present': present}

# ── MAIN ────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multilabel Image Classification Inference')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, default='best_model.pth', help='Path to model weights')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold')
    args = parser.parse_args()

    predict(args.image, args.model, args.threshold)
