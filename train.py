"""
AIMonk Multilabel Classification — Training Script
===================================================
Architecture: ResNet-34 (partially frozen, ImageNet pretrained)
Loss: Masked Weighted BCE (handles missing labels + class imbalance)
Post-Training: Per-attribute threshold tuning on validation set

Usage: python train.py
Output: best_model.pth + loss_curve.png in ./output/
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, f1_score
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
CONFIG = {
    'data_dir': './Multilabel/',
    'image_dir': './Multilabel/images/',
    'label_file': './Multilabel/labels.txt',
    'save_dir': './output/',
    'batch_size': 32,
    'num_epochs': 8,
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'dropout': 0.4,
    'num_workers': 2,
    'image_size': 224,
    'seed': 42,
    'val_split': 0.15,
    'num_attributes': 4,
}

ATTR_NAMES = ['Attribute 1', 'Attribute 2', 'Attribute 3', 'Attribute 4']


# ══════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════
def set_seed(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_single_label(value):
    """
    Parse a single label value from the labels file.
    Handles: '1', '0', 'NA', '1mobilenet' (dirty data), etc.
    
    Returns: (target, mask)
        target: 0.0 or 1.0
        mask: 1.0 (valid) or 0.0 (NA/unknown — ignore in loss)
    """
    val = str(value).strip()

    # Exact matches first
    if val in ['1', '1.0']:
        return 1.0, 1.0
    elif val in ['0', '0.0']:
        return 0.0, 1.0
    elif val.upper() == 'NA':
        return 0.0, 0.0

    # Dirty data: '1mobilenet', '1abc', etc. → treat as 1
    elif val.startswith('1'):
        return 1.0, 1.0

    # Dirty data: '0xxx' → treat as 0
    elif val.startswith('0'):
        return 0.0, 1.0

    # Completely unrecognizable → treat as NA
    else:
        print(f"  ⚠ Unknown label value: '{val}' → treated as NA")
        return 0.0, 0.0


def parse_labels_file(label_file):
    """
    Parse labels.txt — auto-detects format:
      Format A: Tab/space-separated (one row per image)
      Format B: One value per line (5 lines per image)
    
    Returns: DataFrame with columns [filename, attr1, attr2, attr3, attr4]
    """
    with open(label_file, 'r') as f:
        raw_lines = [line.strip() for line in f.readlines()]

    # Remove empty lines
    lines = [l for l in raw_lines if l.strip()]

    # Skip header lines (Image Name, Attr1, Attr2, etc.)
    header_keywords = ['image name', 'attr1', 'attr2', 'attr3', 'attr4',
                       'image_name', 'filename']
    data_start = 0
    for i, line in enumerate(lines):
        if any(kw in line.lower() for kw in header_keywords):
            data_start = i + 1
        else:
            break

    data_lines = lines[data_start:]

    # Auto-detect format
    # Check if first data line contains tabs/multiple spaces → Format A
    first_line = data_lines[0] if data_lines else ''
    parts = first_line.split('\t')
    if len(parts) < 3:
        parts = first_line.split()

    if len(parts) >= 5 and parts[0].endswith(('.jpg', '.png', '.jpeg')):
        # Format A: Tab/space-separated
        print("  📋 Detected label format: tab/space-separated")
        records = []
        for line in data_lines:
            cols = line.split('\t')
            if len(cols) < 5:
                cols = line.split()
            if len(cols) >= 5 and cols[0].endswith(('.jpg', '.png', '.jpeg')):
                records.append(cols[:5])
        df = pd.DataFrame(records, columns=['filename', 'attr1', 'attr2', 'attr3', 'attr4'])

    else:
        # Format B: One value per line (5 lines per image)
        print("  📋 Detected label format: one-value-per-line")
        records = []
        i = 0
        while i < len(data_lines):
            # Find next image filename
            if data_lines[i].endswith(('.jpg', '.png', '.jpeg')):
                fname = data_lines[i]
                attrs = []
                for j in range(1, 5):
                    if i + j < len(data_lines):
                        attrs.append(data_lines[i + j])
                    else:
                        attrs.append('NA')
                records.append([fname] + attrs)
                i += 5
            else:
                i += 1
        df = pd.DataFrame(records, columns=['filename', 'attr1', 'attr2', 'attr3', 'attr4'])

    print(f"  📋 Parsed {len(df)} image entries from labels file")
    return df


def build_labels_and_masks(df, image_dir):
    """
    Convert DataFrame to numpy arrays of labels and masks.
    Also validates that image files exist.
    
    Returns: filenames, labels_array, mask_array, missing_files
    """
    attr_cols = ['attr1', 'attr2', 'attr3', 'attr4']
    num_attrs = len(attr_cols)

    # Check for missing image files
    missing = []
    valid_indices = []
    for idx in range(len(df)):
        fpath = os.path.join(image_dir, df['filename'].iloc[idx])
        if os.path.exists(fpath):
            valid_indices.append(idx)
        else:
            missing.append(df['filename'].iloc[idx])

    if missing:
        print(f"  ⚠ Missing {len(missing)} images: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    df_valid = df.iloc[valid_indices].reset_index(drop=True)

    # Parse labels and masks
    labels_array = np.zeros((len(df_valid), num_attrs), dtype=np.float32)
    mask_array = np.zeros((len(df_valid), num_attrs), dtype=np.float32)

    for i, col in enumerate(attr_cols):
        for j in range(len(df_valid)):
            target, mask = parse_single_label(df_valid[col].iloc[j])
            labels_array[j, i] = target
            mask_array[j, i] = mask

    filenames = df_valid['filename'].values
    return filenames, labels_array, mask_array, missing


def compute_pos_weights(labels_array, mask_array):
    """
    Compute per-attribute positive weights for imbalanced data.
    weight = num_negative / num_positive
    """
    num_attrs = labels_array.shape[1]
    pos_weights = []

    print("\n  📊 Dataset Statistics:")
    print(f"  {'Attr':<8} {'Positive':>8} {'Negative':>8} {'NA':>8} {'Pos Rate':>10} {'Weight':>8}")
    print(f"  {'─' * 56}")

    for i in range(num_attrs):
        valid = mask_array[:, i] == 1
        total_valid = valid.sum()
        pos = labels_array[valid, i].sum()
        neg = total_valid - pos
        na = len(labels_array) - total_valid

        weight = neg / pos if pos > 0 else 1.0
        pos_rate = pos / total_valid * 100 if total_valid > 0 else 0

        pos_weights.append(weight)
        print(f"  Attr{i+1:<4} {int(pos):>8} {int(neg):>8} {int(na):>8} {pos_rate:>9.1f}% {weight:>8.3f}")

    return pos_weights


# ══════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════
class MultilabelDataset(Dataset):
    """
    PyTorch Dataset for multilabel classification with NA handling.
    Returns: (image_tensor, labels_tensor, mask_tensor)
    """

    def __init__(self, filenames, labels, masks, image_dir, transform=None):
        self.filenames = filenames
        self.labels = labels
        self.masks = masks
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.filenames[idx])
        img = np.array(Image.open(img_path).convert('RGB'))

        # Apply transforms
        if self.transform:
            img = self.transform(image=img)['image']

        # Convert labels and masks to tensors
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        mask = torch.tensor(self.masks[idx], dtype=torch.float32)

        return img, label, mask


def get_transforms(image_size, is_train=True):
    """Get augmentation pipeline."""
    if is_train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Rotate(limit=20, p=0.3),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(var_limit=(10, 50), p=0.2),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=15, p=0.3),
            A.CoarseDropout(max_holes=4, max_height=32, max_width=32, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


# ══════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════
class MultilabelResNet34(nn.Module):
    """
    ResNet-34 with partially frozen layers for multilabel classification.
    Frozen: conv1, bn1, layer1, layer2 (generic low-level features)
    Trainable: layer3, layer4, fc (domain-specific features)
    """

    def __init__(self, num_classes=4, dropout=0.4, pretrained=True):
        super().__init__()

        # Load backbone
        if pretrained:
            self.backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            print("  🏗 Loaded ResNet-34 with ImageNet pretrained weights")
        else:
            self.backbone = models.resnet34(weights=None)
            print("  🏗 Loaded ResNet-34 without pretrained weights")

        # Freeze early layers
        frozen_prefixes = ['conv1', 'bn1', 'layer1', 'layer2']
        frozen_count = 0
        for name, param in self.backbone.named_parameters():
            if any(name.startswith(prefix) for prefix in frozen_prefixes):
                param.requires_grad = False
                frozen_count += 1

        # Replace FC head: 512 → num_classes
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

        # Count parameters
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  🏗 Parameters: {total:,} total | {trainable:,} trainable | "
              f"{total - trainable:,} frozen")

    def forward(self, x):
        return self.backbone(x)


# ══════════════════════════════════════════════════════════════
# LOSS FUNCTION
# ══════════════════════════════════════════════════════════════
class MaskedWeightedBCELoss(nn.Module):
    """
    Binary Cross-Entropy loss that:
    1. Masks out NA labels (mask=0 → no gradient contribution)
    2. Applies per-attribute pos_weight for class imbalance
    """

    def __init__(self, pos_weight):
        super().__init__()
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, logits, targets, masks):
        # Per-element BCE (no reduction)
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='none'
        )

        # Mask: NA elements contribute 0 to loss
        masked_loss = bce * masks

        # Average only over valid (non-NA) elements
        valid_count = masks.sum()
        if valid_count > 0:
            return masked_loss.sum() / valid_count
        return masked_loss.sum()


# ══════════════════════════════════════════════════════════════
# TRAINING FUNCTIONS
# ══════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch. Returns list of (iteration, loss) tuples."""
    model.train()
    iter_losses = []

    for images, labels, masks in loader:
        images = images.to(device)
        labels = labels.to(device)
        masks = masks.to(device)

        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels, masks)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iter_losses.append(loss.item())

    return iter_losses


def validate(model, loader, criterion, device, thresholds=None):
    """
    Validate model. Returns val_loss, per-attribute metrics, 
    and raw predictions for threshold tuning.
    """
    if thresholds is None:
        thresholds = [0.5] * 4

    model.eval()
    all_preds, all_labels, all_masks = [], [], []
    total_loss = 0

    with torch.no_grad():
        for images, labels, masks in loader:
            images = images.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels, masks)
            total_loss += loss.item()

            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(labels.cpu().numpy())
            all_masks.append(masks.cpu().numpy())

    avg_loss = total_loss / len(loader)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_masks = np.concatenate(all_masks)

    # Per-attribute metrics
    attr_metrics = []
    for i in range(4):
        valid = all_masks[:, i] == 1
        if valid.sum() > 0:
            binary = (all_preds[valid, i] >= thresholds[i]).astype(int)
            p, r, f1, _ = precision_recall_fscore_support(
                all_labels[valid, i], binary, average='binary', zero_division=0
            )
            attr_metrics.append({'precision': p, 'recall': r, 'f1': f1})
        else:
            attr_metrics.append({'precision': 0, 'recall': 0, 'f1': 0})

    return avg_loss, attr_metrics, all_preds, all_labels, all_masks


def tune_thresholds(all_preds, all_labels, all_masks):
    """
    Grid search for optimal per-attribute classification thresholds.
    Optimizes F1-score independently for each attribute.
    """
    num_attrs = all_preds.shape[1]
    best_thresholds = []

    print("\n  🔍 Tuning per-attribute thresholds...")
    print(f"  {'Attr':<8} {'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'─' * 50}")

    for i in range(num_attrs):
        valid = all_masks[:, i] == 1
        if valid.sum() == 0:
            best_thresholds.append(0.5)
            continue

        preds_i = all_preds[valid, i]
        labels_i = all_labels[valid, i]

        best_f1 = 0
        best_t = 0.5
        best_p = 0
        best_r = 0

        for t in np.arange(0.05, 0.95, 0.05):
            binary = (preds_i >= t).astype(int)
            p, r, f1, _ = precision_recall_fscore_support(
                labels_i, binary, average='binary', zero_division=0
            )
            if f1 > best_f1:
                best_f1 = f1
                best_t = round(t, 2)
                best_p = p
                best_r = r

        best_thresholds.append(best_t)
        print(f"  Attr{i+1:<4} {best_t:>10.2f} {best_p:>10.3f} {best_r:>10.3f} {best_f1:>10.3f}")

    return best_thresholds


# ══════════════════════════════════════════════════════════════
# LOSS CURVE PLOTTING
# ══════════════════════════════════════════════════════════════
def save_loss_curve(all_iter_losses, epoch_train_losses, epoch_val_losses, save_dir):
    """
    Save two versions of loss curves:
    1. loss_curve.png — Standalone plot meeting exact assignment specs
    2. loss_curve_detailed.png — Two-panel with train vs val comparison
    """

    # ── Plot 1: Standalone (meets exact assignment spec) ──
    iters = list(range(1, len(all_iter_losses) + 1))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(iters, all_iter_losses, alpha=0.3, color='blue', label='Raw')

    # Smoothed curve
    window = max(10, len(all_iter_losses) // 50)
    if len(all_iter_losses) > window:
        smoothed = np.convolve(all_iter_losses, np.ones(window) / window, mode='valid')
        smooth_iters = list(range(window, len(all_iter_losses) + 1))
        ax.plot(smooth_iters, smoothed, color='red', linewidth=2, label='Smoothed')

    ax.set_xlabel('iteration_number')
    ax.set_ylabel('training_loss')
    ax.set_title('Aimonk_multilabel_problem')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'loss_curve.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  ✅ loss_curve.png saved (assignment spec)")

    # ── Plot 2: Detailed two-panel ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Aimonk_multilabel_problem', fontsize=14, fontweight='bold')

    # Left: Per-iteration training loss
    ax1.set_title('Training Loss (per iteration)')
    ax1.plot(iters, all_iter_losses, alpha=0.3, color='blue', label='Raw')
    if len(all_iter_losses) > window:
        ax1.plot(smooth_iters, smoothed, color='red', linewidth=2, label='Smoothed')
    ax1.set_xlabel('iteration_number')
    ax1.set_ylabel('training_loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Per-epoch train vs val loss
    epochs = list(range(1, len(epoch_train_losses) + 1))
    ax2.set_title('Train vs Validation Loss')
    ax2.plot(epochs, epoch_train_losses, 'b-o', label='Train Loss')
    ax2.plot(epochs, epoch_val_losses, 'r-o', label='Val Loss')

    # Mark best epoch
    best_epoch = np.argmin(epoch_val_losses) + 1
    ax2.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7,
                label=f'Best epoch ({best_epoch})')

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'loss_curve_detailed.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  ✅ loss_curve_detailed.png saved (detailed version)")


# ══════════════════════════════════════════════════════════════
# MAIN TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  AIMonk Multilabel Classification — Training")
    print("=" * 60)

    # ── Setup ──
    set_seed(CONFIG['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")
    os.makedirs(CONFIG['save_dir'], exist_ok=True)

    # ── Parse Labels ──
    print("\n📋 Parsing labels...")
    df = parse_labels_file(CONFIG['label_file'])
    filenames, labels_array, mask_array, missing = build_labels_and_masks(
        df, CONFIG['image_dir']
    )
    print(f"  ✅ {len(filenames)} valid images loaded")

    # ── Compute Class Weights ──
    pos_weights = compute_pos_weights(labels_array, mask_array)
    pos_weight_tensor = torch.tensor(pos_weights, dtype=torch.float32).to(device)

    # ── Train/Val Split (stratified on Attr4) ──
    print("\n📊 Splitting dataset...")
    # Create stratification groups: 0=NA, 1=negative, 2=positive for Attr4
    stratify_col = np.where(
        mask_array[:, 3] == 0, 0,
        np.where(labels_array[:, 3] == 1, 2, 1)
    )

    train_idx, val_idx = train_test_split(
        np.arange(len(filenames)),
        test_size=CONFIG['val_split'],
        random_state=CONFIG['seed'],
        stratify=stratify_col
    )
    print(f"  Train: {len(train_idx)} | Val: {len(val_idx)}")

    # ── Datasets & Loaders ──
    train_transform = get_transforms(CONFIG['image_size'], is_train=True)
    val_transform = get_transforms(CONFIG['image_size'], is_train=False)

    train_ds = MultilabelDataset(
        filenames[train_idx], labels_array[train_idx], mask_array[train_idx],
        CONFIG['image_dir'], train_transform
    )
    val_ds = MultilabelDataset(
        filenames[val_idx], labels_array[val_idx], mask_array[val_idx],
        CONFIG['image_dir'], val_transform
    )

    train_loader = DataLoader(
        train_ds, batch_size=CONFIG['batch_size'], shuffle=True,
        num_workers=CONFIG['num_workers'], pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=CONFIG['batch_size'], shuffle=False,
        num_workers=CONFIG['num_workers'], pin_memory=True
    )

    # ── Model, Loss, Optimizer ──
    print("\n🏗 Building model...")
    model = MultilabelResNet34(
        num_classes=CONFIG['num_attributes'],
        dropout=CONFIG['dropout'],
        pretrained=True
    ).to(device)

    criterion = MaskedWeightedBCELoss(pos_weight_tensor).to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG['lr'],
        weight_decay=CONFIG['weight_decay']
    )

    # ── Training Loop ──
    print("\n🚀 Starting training...")
    print(f"  Epochs: {CONFIG['num_epochs']} | Batch size: {CONFIG['batch_size']} | "
          f"LR: {CONFIG['lr']}")
    print(f"  {'─' * 60}")

    best_f1 = 0
    best_epoch = 0
    all_iter_losses = []
    epoch_train_losses = []
    epoch_val_losses = []

    for epoch in range(CONFIG['num_epochs']):
        # Train
        iter_losses = train_one_epoch(model, train_loader, criterion, optimizer, device)
        all_iter_losses.extend(iter_losses)
        avg_train_loss = np.mean(iter_losses)
        epoch_train_losses.append(avg_train_loss)

        # Validate
        avg_val_loss, attr_metrics, val_preds, val_labels, val_masks = validate(
            model, val_loader, criterion, device
        )
        epoch_val_losses.append(avg_val_loss)

        # Compute average F1
        f1_scores = [m['f1'] for m in attr_metrics]
        avg_f1 = np.mean(f1_scores)

        # Print epoch summary
        print(f"\n  Epoch {epoch + 1}/{CONFIG['num_epochs']}")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Avg F1: {avg_f1:.4f}")

        for i, m in enumerate(attr_metrics):
            print(f"    Attr{i+1}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")

        # Save best model
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_epoch = epoch + 1

            torch.save({
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'avg_f1': best_f1,
                'attr_metrics': attr_metrics,
                'pos_weights': pos_weights,
                'config': CONFIG,
            }, os.path.join(CONFIG['save_dir'], 'best_model.pth'))

            print(f"  ★ New best model saved! Avg F1={best_f1:.4f}")

    print(f"\n{'═' * 60}")
    print(f"  Training complete! Best Avg F1: {best_f1:.4f} at epoch {best_epoch}")
    print(f"{'═' * 60}")

    # ── Threshold Tuning ──
    print("\n🔍 Per-attribute threshold tuning on validation set...")

    # Load best model
    best_ckpt = torch.load(os.path.join(CONFIG['save_dir'], 'best_model.pth'),
                           map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])

    # Get validation predictions from best model
    _, _, val_preds, val_labels, val_masks = validate(
        model, val_loader, criterion, device
    )

    # Find optimal thresholds
    best_thresholds = tune_thresholds(val_preds, val_labels, val_masks)
    print(f"\n  ✅ Tuned thresholds: {best_thresholds}")

    # Re-evaluate with tuned thresholds
    print("\n  📊 Results with tuned thresholds:")
    _, tuned_metrics, _, _, _ = validate(
        model, val_loader, criterion, device, thresholds=best_thresholds
    )

    tuned_f1s = []
    print(f"  {'Attr':<8} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Threshold':>10}")
    print(f"  {'─' * 50}")
    for i, m in enumerate(tuned_metrics):
        print(f"  Attr{i+1:<4} {m['precision']:>10.3f} {m['recall']:>10.3f} "
              f"{m['f1']:>10.3f} {best_thresholds[i]:>10.2f}")
        tuned_f1s.append(m['f1'])

    tuned_avg_f1 = np.mean(tuned_f1s)
    print(f"\n  Avg F1 (default 0.5): {best_f1:.4f}")
    print(f"  Avg F1 (tuned):       {tuned_avg_f1:.4f}")
    print(f"  Improvement:          {tuned_avg_f1 - best_f1:+.4f}")

    # Save model with thresholds
    best_ckpt['thresholds'] = best_thresholds
    best_ckpt['tuned_avg_f1'] = tuned_avg_f1
    best_ckpt['tuned_metrics'] = tuned_metrics
    torch.save(best_ckpt, os.path.join(CONFIG['save_dir'], 'best_model.pth'))
    print("  ✅ Model re-saved with tuned thresholds!")

    # ── Save Loss Curves ──
    print("\n📈 Saving loss curves...")
    save_loss_curve(all_iter_losses, epoch_train_losses, epoch_val_losses, CONFIG['save_dir'])

    # ── Final Summary ──
    print(f"\n{'═' * 60}")
    print(f"  📦 Deliverables saved to {CONFIG['save_dir']}:")
    print(f"     • best_model.pth (weights + thresholds)")
    print(f"     • loss_curve.png (assignment spec)")
    print(f"     • loss_curve_detailed.png (train vs val)")
    print(f"{'═' * 60}")


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    main()