"""
=============================================================================
SignalScope - High-Resolution ViT-B/16 Fine-Tuning Pipeline
Target: Modern Real-World Photography vs. Diffusion Synthetics (Midjourney / SD)
Hardware: Google Colab (Free Tesla T4 GPU, ~8 minutes)
Benchmark Citing: Section 4.1 & 4.2 (GenImage / HuggingFace Public Data)
=============================================================================
"""

# Cell 1: Install Dependencies
# !pip install -q datasets torchvision scikit-learn pillow

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from datasets import load_dataset
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import numpy as np

# 1. Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using compute device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# 2. Dataset Loading (Parveshiiii/AI-vs-Real: 14,000 modern 512px images)
print("\n[1/5] Streaming modern Real vs. AI dataset from Hugging Face...")
raw_dataset = load_dataset("Parveshiiii/AI-vs-Real", split="train")
print(f"Total samples loaded: {len(raw_dataset)}")

# Split into 80% train / 20% validation
split_data = raw_dataset.train_test_split(test_size=0.20, seed=42)
train_raw = split_data["train"]
val_raw = split_data["test"]
print(f"Training set: {len(train_raw)} | Validation set: {len(val_raw)}")

# 3. Data Transformations with Data Augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class HFImageDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item["image"].convert("RGB")
        # In Parveshiiii/AI-vs-Real:
        # Note: Class 0 = FAKE, Class 1 = REAL (matching our SignalScope convention)
        label = int(item["binary_label"])
        if self.transform:
            img = self.transform(img)
        return img, label

train_loader = DataLoader(HFImageDataset(train_raw, train_transform), batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(HFImageDataset(val_raw, val_transform), batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

# 4. Model Architecture: ViT-B/16
print("\n[2/5] Initializing Vision Transformer (ViT-B/16)...")
model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
# Replace classification head with 2 classes (0: Fake, 1: Real)
in_features = model.heads.head.in_features
model.heads.head = nn.Linear(in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-2)
scaler = torch.cuda.amp.GradScaler()

# 5. Training Loop (2 Epochs with Mixed Precision FP16)
epochs = 2
print(f"\n[3/5] Starting fine-tuning for {epochs} epochs on modern images...")
start_time = time.time()

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (i + 1) % 50 == 0 or (i + 1) == len(train_loader):
            print(f"Epoch [{epoch+1}/{epochs}] Step [{i+1}/{len(train_loader)}] Loss: {loss.item():.4f} Acc: {correct/total:.4f}")

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f"--> Epoch {epoch+1} Complete | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")

total_duration = time.time() - start_time
print(f"Training completed in {total_duration/60:.2f} minutes!")

# 6. Validation & Official Hackathon Metrics
print("\n[4/5] Evaluating on Validation Set (ROC-AUC & Macro-F1)...")
model.eval()
all_preds = []
all_probs = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        with torch.cuda.amp.autocast():
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1] # Probability of REAL (Class 1)
            _, preds = torch.max(outputs, 1)
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

auc = roc_auc_score(all_labels, all_probs)
f1 = f1_score(all_labels, all_preds, average="macro")
cm = confusion_matrix(all_labels, all_preds)

print("\n" + "="*50)
print("  OFFICIAL VALIDATION BENCHMARK RESULTS")
print("="*50)
print(f"  ROC-AUC Score:  {auc:.4f}")
print(f"  Macro-F1 Score: {f1:.4f}")
print(f"  Confusion Matrix:\n{cm}")
print("="*50)

# 7. Save and Download Weights
output_filename = "vit_b16_signalscope.pth"
torch.save(model.state_dict(), output_filename)
print(f"\n[5/5] Model weights saved to: {output_filename} ({os.path.getsize(output_filename)/(1024*1024):.1f} MB)")

# Trigger Google Colab Download
try:
    from google.colab import files
    files.download(output_filename)
    print("Download triggered automatically in Colab!")
except ImportError:
    print(f"File saved locally as '{output_filename}'.")
