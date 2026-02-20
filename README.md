# AIMonk — Multilabel Image Classification

## Problem
Multilabel binary classification of fashion/clothing images with 4 attributes per image.

## Dataset
- **972 images** (3 missing from original 975)
- **4 binary attributes** per image
- Labels contain **NA values** (missing ground truth) → handled with masked loss
- **Class imbalance**: Attr4 has only 7.7% positive samples

| Attribute | Positive | Negative | NA  | Pos Rate |
|-----------|----------|----------|-----|----------|
| Attr1     | 789      | 106      | 80  | 88.2%    |
| Attr2     | 713      | 171      | 91  | 80.7%    |
| Attr3     | 466      | 416      | 93  | 52.8%    |
| Attr4     | 68       | 813      | 94  | 7.7%     |

## Approach

### Architecture
- **ResNet-34** (ImageNet pretrained)
- **Partially frozen**: conv1, bn1, layer1, layer2 (frozen) | layer3, layer4, FC (trainable)
- **Head**: Dropout(0.4) → Linear(512, 4)

### Training Strategy
- **Loss**: Masked Weighted BCE — handles missing labels (NA) with binary masks and class imbalance with per-attribute pos_weights
- **Optimizer**: Adam (lr=1e-4, weight_decay=1e-4)
- **Augmentation**: HorizontalFlip, VerticalFlip, Rotate, ColorJitter, GaussianBlur, GaussNoise, ShiftScaleRotate, CoarseDropout
- **Split**: 85/15 stratified on Attr4
- **Epochs**: 8 (best at epoch 7)

### Results (Validation Set)

| Attribute   | Precision | Recall | F1-Score |
|-------------|-----------|--------|----------|
| Attr1       | 0.90      | 0.81   | 0.85     |
| Attr2       | 0.83      | 0.76   | 0.79     |
| Attr3       | 0.54      | 0.60   | 0.57     |
| Attr4       | 0.00      | 0.00   | 0.00     |
| **Average** | —         | —      | **0.553** |

> **Note on Attr4**: With only 68 positive samples (7.7%), the model struggles to learn this attribute. The pos_weight of 12.0 helps but is insufficient given the extreme imbalance. Focal loss or aggressive oversampling could improve this.

## Files

```
├── train.py                # Complete training pipeline
├── inference.py            # Standalone inference script
├── best_model.pth          # Trained model weights (epoch 7)
├── loss_curve.png          # Training loss curve
├── sample_predictions.png  # Visualization of predictions
├── README.md               # This file
└── notebook.ipynb          # Full Colab notebook
```

## Usage

### Training
```bash
# Place dataset in ./Multilabel/ directory
python train.py
```
Output: `best_model.pth` and `loss_curve.png` saved to `./output/`

### Inference
```bash
python inference.py --image path/to/image.jpg --model best_model.pth
python inference.py --image path/to/image.jpg --model best_model.pth --threshold 0.3
```

### Example Output
```text
==================================================
  IMAGE: image_0.jpg
  THRESHOLD: 0.5
──────────────────────────────────────────────────
  ✓ Attribute 1: PRESENT  (confidence: 61.9%)
  ✗ Attribute 2: ABSENT   (confidence: 36.5%)
  ✗ Attribute 3: ABSENT   (confidence: 16.3%)
  ✓ Attribute 4: PRESENT  (confidence: 94.7%)
──────────────────────────────────────────────────
  Attributes present: ['Attribute 1', 'Attribute 4']
==================================================
```

## Requirements
```text
torch>=1.12
torchvision>=0.13
albumentations>=1.3
Pillow
numpy
pandas
scikit-learn
matplotlib
```

## Experiments Summary

| # | Model          | Key Change               | Best Epoch | Avg F1 |
|---|----------------|--------------------------|------------|--------|
| 1 | ResNet-34 v1   | Simple baseline          | 2          | 0.567  |
| 2 | ResNet-34 v2   | Frozen layer3            | —          | 0.403  |
| 3 | ResNet-18      | Full training            | 36         | 0.504  |
| 4 | ResNet-34      | Low LR + tricks          | 1          | 0.405  |
| 5 | ResNet-34 (prod) | Strong augmentation    | 7          | 0.553  |

## Key Design Decisions

1. **Masked Loss**: NA labels are masked out during loss computation so they don't corrupt gradients
2. **Weighted BCE**: Per-attribute pos_weights compensate for class imbalance
3. **Partial Freezing**: Early ResNet layers (conv1→layer2) are frozen to prevent overfitting on small dataset; deeper layers (layer3→layer4) are fine-tuned
4. **Strong Augmentation**: Aggressive data augmentation delays overfitting (best epoch moved from 2 to 7)

## Author
AIMonk Assignment Submission
