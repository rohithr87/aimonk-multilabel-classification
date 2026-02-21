"""
AIMonk Multilabel Classification — Inference Script
====================================================
Supports:
  - Single image prediction
  - Batch prediction (entire folder)
  - Per-attribute tuned thresholds (loaded from model checkpoint)
  - Override with default threshold

Usage:
  python inference.py --image path/to/image.jpg --model best_model.pth
  python inference.py --folder path/to/images/ --model best_model.pth
  python inference.py --image path/to/image.jpg --model best_model.pth --default-threshold 0.3
"""

import argparse
import os
import glob
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ══════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE (must match training)
# ══════════════════════════════════════════════════════════════
class MultilabelResNet34(nn.Module):
    """
    ResNet-34 for multilabel classification.
    weights=None because we load from checkpoint.
    """

    def __init__(self, num_classes=4, dropout=0.4):
        super().__init__()
        self.backbone = models.resnet34(weights=None)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


# ══════════════════════════════════════════════════════════════
# PREPROCESSING (must match validation transform from training)
# ══════════════════════════════════════════════════════════════
TRANSFORM = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

ATTR_NAMES = ['Attribute 1', 'Attribute 2', 'Attribute 3', 'Attribute 4']


# ══════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════
def load_model(model_path, device):
    """
    Load model and thresholds from checkpoint.
    Returns: model, thresholds, checkpoint_info
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Extract model config from checkpoint if available
    num_classes = checkpoint.get('num_attributes', 4)
    dropout = checkpoint.get('dropout', 0.4)
    if 'config' in checkpoint:
        num_classes = checkpoint['config'].get('num_attributes', num_classes)
        dropout = checkpoint['config'].get('dropout', dropout)

    # Build and load model
    model = MultilabelResNet34(num_classes=num_classes, dropout=dropout).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load thresholds
    thresholds = checkpoint.get('thresholds', None)

    # Checkpoint info
    info = {
        'epoch': checkpoint.get('epoch', '?'),
        'avg_f1': checkpoint.get('avg_f1', '?'),
        'tuned_avg_f1': checkpoint.get('tuned_avg_f1', '?'),
        'has_tuned_thresholds': thresholds is not None,
    }

    return model, thresholds, info


# ══════════════════════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════════════════════
def predict_single(image_path, model, thresholds, device):
    """
    Predict attributes for a single image.
    Returns dict with probabilities, binary predictions, and present attributes.
    """
    # Load and preprocess
    img = np.array(Image.open(image_path).convert('RGB'))
    input_tensor = TRANSFORM(image=img)['image'].unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    # Apply per-attribute thresholds
    binary = np.array([1 if p >= t else 0 for p, t in zip(probs, thresholds)])
    present = [ATTR_NAMES[i] for i in range(len(binary)) if binary[i] == 1]

    return {
        'probabilities': probs,
        'binary': binary,
        'present': present,
        'thresholds': thresholds,
    }


def print_prediction(image_path, result):
    """Pretty-print prediction result."""
    fname = os.path.basename(image_path)
    thresholds = result['thresholds']

    print(f"\n{'=' * 60}")
    print(f"  IMAGE: {fname}")
    print(f"  THRESHOLDS: {[f'{t:.2f}' for t in thresholds]}")
    print(f"{'─' * 60}")

    for i, (name, prob, pred, thresh) in enumerate(
            zip(ATTR_NAMES, result['probabilities'], result['binary'], thresholds)):
        status = "PRESENT" if pred == 1 else "ABSENT"
        symbol = "✓" if pred == 1 else "✗"
        print(f"  {symbol} {name}: {status:<8} (conf: {prob * 100:5.1f}%, thresh: {thresh:.2f})")

    print(f"{'─' * 60}")
    if result['present']:
        print(f"  Attributes present: {result['present']}")
    else:
        print(f"  No attributes detected.")
    print(f"{'=' * 60}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description='AIMonk Multilabel Image Classification — Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference.py --image photo.jpg --model best_model.pth
  python inference.py --folder ./images/ --model best_model.pth
  python inference.py --image photo.jpg --model best_model.pth --default-threshold 0.3
        """
    )

    # Input: single image or folder
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', type=str, help='Path to a single image')
    input_group.add_argument('--folder', type=str, help='Path to folder of images')

    # Model
    parser.add_argument('--model', type=str, default='best_model.pth',
                        help='Path to model checkpoint (default: best_model.pth)')

    # Threshold override
    parser.add_argument('--default-threshold', type=float, default=None,
                        help='Override with single threshold for all attributes '
                             '(default: use tuned thresholds from checkpoint)')

    args = parser.parse_args()

    # ── Setup ──
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load Model ──
    print(f"Loading model from: {args.model}")
    model, tuned_thresholds, info = load_model(args.model, device)

    print(f"  Epoch: {info['epoch']} | F1: {info.get('tuned_avg_f1', info.get('avg_f1', '?'))}")

    # ── Determine Thresholds ──
    if args.default_threshold is not None:
        # User explicitly overrides
        thresholds = [args.default_threshold] * 4
        print(f"  Using override threshold: {args.default_threshold} for all attributes")
    elif tuned_thresholds is not None:
        # Use tuned thresholds from checkpoint
        thresholds = tuned_thresholds
        print(f"  Using tuned thresholds: {[f'{t:.2f}' for t in thresholds]}")
    else:
        # Fallback
        thresholds = [0.5] * 4
        print(f"  No tuned thresholds found — using default 0.5")

    # ── Single Image ──
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image not found: {args.image}")
            return

        result = predict_single(args.image, model, thresholds, device)
        print_prediction(args.image, result)

    # ── Batch Folder ──
    elif args.folder:
        if not os.path.isdir(args.folder):
            print(f"Error: Folder not found: {args.folder}")
            return

        # Find all images
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(args.folder, ext)))
            image_paths.extend(glob.glob(os.path.join(args.folder, ext.upper())))

        image_paths = sorted(set(image_paths))
        print(f"\n  Found {len(image_paths)} images in {args.folder}")

        if len(image_paths) == 0:
            print("  No images found.")
            return

        # Predict all
        all_results = []
        for img_path in image_paths:
            result = predict_single(img_path, model, thresholds, device)
            print_prediction(img_path, result)
            all_results.append((img_path, result))

        # Summary
        print(f"\n{'═' * 60}")
        print(f"  BATCH SUMMARY: {len(image_paths)} images processed")
        print(f"{'─' * 60}")

        # Count prediction patterns
        patterns = {}
        for _, result in all_results:
            pattern = str(result['binary'].tolist())
            patterns[pattern] = patterns.get(pattern, 0) + 1

        print(f"  Prediction patterns:")
        for pattern, count in sorted(patterns.items(), key=lambda x: -x[1]):
            pct = count / len(all_results) * 100
            print(f"    {pattern}: {count} ({pct:.1f}%)")

        print(f"{'═' * 60}")


if __name__ == '__main__':
    main()