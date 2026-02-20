"""
AIMonk Multilabel Classification — Training Script
===================================================
Architecture: ResNet-34 (partially frozen, ImageNet pretrained)
Loss: Masked Weighted BCE (handles missing labels + class imbalance)

Usage: python train.py
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
from sklearn.metrics import precision_recall_fscore_support
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

# ── CONFIG ──────────────────────────────────────────────────
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
}

os.makedirs(CONFIG['save_dir'], exist_ok=True)
torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── PARSE LABELS ────────────────────────────────────────────
df = pd.read_csv(CONFIG['label_file'], sep=r'\s+', header=None,
                 names=['filename', 'attr1', 'attr2', 'attr3', 'attr4'])

# Find missing images
missing = []
for f in df['filename']:
    if not os.path.exists(os.path.join(CONFIG['image_dir'], f)):
        missing.append(f)
print(f"Missing images: {missing}")

df_clean = df[~df['filename'].isin(missing)].reset_index(drop=True)
attr_cols = ['attr1', 'attr2', 'attr3', 'attr4']

# Build labels and masks
labels_array = np.zeros((len(df_clean), 4), dtype=np.float32)
mask_array = np.zeros((len(df_clean), 4), dtype=np.float32)

for i, col in enumerate(attr_cols):
    for j in range(len(df_clean)):
        val = str(df_clean[col].iloc[j]).strip()
        if val in ['0', '0.0']:
            labels_array[j, i] = 0.0
            mask_array[j, i] = 1.0
        elif val in ['1', '1.0']:
            labels_array[j, i] = 1.0
            mask_array[j, i] = 1.0
        else:
            labels_array[j, i] = 0.0
            mask_array[j, i] = 0.0

filenames = df_clean['filename'].values

# Compute pos_weights
pos_weights = []
for i in range(4):
    valid = mask_array[:, i] == 1
    pos = labels_array[valid, i].sum()
    neg = valid.sum() - pos
    w = neg / pos if pos > 0 else 1.0
    pos_weights.append(w)
    print(f"  Attr{i+1}: pos={int(pos)}, neg={int(neg)}, weight={w:.3f}")

pos_weight_tensor = torch.tensor(pos_weights, dtype=torch.float32).to(device)

# ── TRAIN/VAL SPLIT ────────────────────────────────────────
attr4_labels = labels_array[:, 3].copy()
attr4_valid = mask_array[:, 3] == 1
attr4_for_split = np.where(attr4_valid, attr4_labels, -1)
stratify_col = (attr4_for_split * 2 + mask_array[:, 3]).astype(int)

train_idx, val_idx = train_test_split(
    np.arange(len(filenames)), test_size=0.15, random_state=CONFIG['seed'],
    stratify=stratify_col
)
print(f"Train: {len(train_idx)} | Val: {len(val_idx)}")

# ── AUGMENTATION ────────────────────────────────────────────
train_transform = A.Compose([
    A.Resize(CONFIG['image_size'], CONFIG['image_size']),
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

val_transform = A.Compose([
    A.Resize(CONFIG['image_size'], CONFIG['image_size']),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# ── DATASET ─────────────────────────────────────────────────
class MultilabelDataset(Dataset):
    def __init__(self, fnames, labels, masks, img_dir, transform=None):
        self.fnames = fnames
        self.labels = labels
        self.masks = masks
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.fnames[idx])
        img = np.array(Image.open(img_path).convert('RGB'))
        if self.transform:
            img = self.transform(image=img)['image']
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        mask = torch.tensor(self.masks[idx], dtype=torch.float32)
        return img, label, mask

train_ds = MultilabelDataset(filenames[train_idx], labels_array[train_idx],
                              mask_array[train_idx], CONFIG['image_dir'], train_transform)
val_ds = MultilabelDataset(filenames[val_idx], labels_array[val_idx],
                            mask_array[val_idx], CONFIG['image_dir'], val_transform)

train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True,
                          num_workers=CONFIG['num_workers'], pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False,
                        num_workers=CONFIG['num_workers'], pin_memory=True)

# ── LOSS ────────────────────────────────────────────────────
class MaskedWeightedBCELoss(nn.Module):
    def __init__(self, pos_weight):
        super().__init__()
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, logits, targets, masks):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='none'
        )
        masked = bce * masks
        if masks.sum() > 0:
            return masked.sum() / masks.sum()
        return masked.sum()

# ── MODEL ───────────────────────────────────────────────────
class ProductionResNet34(nn.Module):
    def __init__(self, num_classes=4, dropout=0.4):
        super().__init__()
        self.backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        # Freeze early layers
        for name, param in self.backbone.named_parameters():
            if any(name.startswith(prefix) for prefix in ['conv1', 'bn1', 'layer1', 'layer2']):
                param.requires_grad = False

        # Replace FC head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

model = ProductionResNet34(num_classes=4, dropout=CONFIG['dropout']).to(device)
criterion = MaskedWeightedBCELoss(pos_weight_tensor).to(device)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])

# ── TRAINING LOOP ───────────────────────────────────────────
best_f1 = 0
all_iter_losses = []
global_iter = 0

for epoch in range(CONFIG['num_epochs']):
    # Train
    model.train()
    epoch_loss = 0
    for batch_idx, (images, labels, masks) in enumerate(train_loader):
        images, labels, masks = images.to(device), labels.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        global_iter += 1
        all_iter_losses.append((global_iter, loss.item()))

    avg_train_loss = epoch_loss / len(train_loader)

    # Validate
    model.eval()
    all_preds, all_labels, all_masks = [], [], []
    val_loss = 0

    with torch.no_grad():
        for images, labels, masks in val_loader:
            images, labels, masks = images.to(device), labels.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels, masks)
            val_loss += loss.item()

            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_masks.append(masks.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_masks = np.concatenate(all_masks)

    binary_preds = (all_preds > 0.5).astype(int)

    f1_scores = []
    for i in range(4):
        valid = all_masks[:, i] == 1
        if valid.sum() > 0:
            p, r, f1, _ = precision_recall_fscore_support(
                all_labels[valid, i], binary_preds[valid, i], average='binary', zero_division=0
            )
            f1_scores.append(f1)
            print(f"  Attr{i+1}: P={p:.3f} R={r:.3f} F1={f1:.3f}")

    avg_f1 = np.mean(f1_scores) if f1_scores else 0
    print(f"Epoch {epoch+1}/{CONFIG['num_epochs']} | "
          f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Avg F1: {avg_f1:.4f}")

    if avg_f1 > best_f1:
        best_f1 = avg_f1
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'val_loss': avg_val_loss,
            'avg_f1': best_f1,
            'pos_weights': pos_weights,
            'model_name': 'resnet34',
            'num_attributes': 4,
            'dropout': CONFIG['dropout'],
            'image_size': CONFIG['image_size'],
            'imagenet_mean': [0.485, 0.456, 0.406],
            'imagenet_std': [0.229, 0.224, 0.225],
        }, os.path.join(CONFIG['save_dir'], 'best_model.pth'))
        print(f"  ★ New best model saved! F1={best_f1:.4f}")

print(f"\n✅ Training complete! Best F1: {best_f1:.4f}")

# ── SAVE LOSS CURVE ─────────────────────────────────────────
iters = [x[0] for x in all_iter_losses]
losses = [x[1] for x in all_iter_losses]

plt.figure(figsize=(10, 6))
plt.plot(iters, losses, alpha=0.3, color='blue', label='Raw')

window = 10
if len(losses) > window:
    smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
    plt.plot(range(window, len(losses)+1), smoothed, color='red', linewidth=2, label='Smoothed')

plt.xlabel('iteration_number')
plt.ylabel('training_loss')
plt.title('Aimonk_multilabel_problem')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(CONFIG['save_dir'], 'loss_curve.png'), dpi=150, bbox_inches='tight')
print("✅ Loss curve saved!")
