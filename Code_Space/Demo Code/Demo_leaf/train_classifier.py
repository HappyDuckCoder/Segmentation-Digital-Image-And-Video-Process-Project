# ==========================================
# Train Linear Classifier on SSL Backbone
# ==========================================

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from tqdm import tqdm
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)

NUM_CLASSES = 5
SSL_CKPT = "checkpoint_ssl.pth"

# ============================
# Load SSL backbone (student)
# ============================
print("🔄 Loading SSL backbone...")

ckpt = torch.load(SSL_CKPT, map_location="cpu")

backbone = timm.create_model(
    "vit_small_patch16_224",
    pretrained=False,
    num_classes=0
)

backbone.load_state_dict(ckpt["student"])
backbone.to(device)
backbone.eval()

for p in backbone.parameters():
    p.requires_grad = False

print("✅ SSL Backbone Loaded")


# ============================
# Linear classifier head
# ============================
classifier = nn.Linear(backbone.num_features, NUM_CLASSES).to(device)

optimizer = torch.optim.AdamW(classifier.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()

# ============================
# Dataset
# ============================
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225)),
])

train_ds = datasets.ImageFolder("dataset/train", tf)
val_ds   = datasets.ImageFolder("dataset/val", tf)

train_ld = DataLoader(train_ds, batch_size=32, shuffle=True)
val_ld   = DataLoader(val_ds, batch_size=32)

# ============================
# Training loop
# ============================
EPOCHS = 30

for epoch in range(EPOCHS):

    t0 = time.time()

    classifier.train()
    total_loss = 0

    pbar = tqdm(train_ld,
                desc=f"Epoch {epoch+1}/{EPOCHS}",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, loss={postfix}]")

    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)

        with torch.no_grad():
            feats = backbone(imgs)

        logits = classifier(feats)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    # ============================
    # Validation
    # ============================
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in val_ld:
            imgs, labels = imgs.to(device), labels.to(device)
            feats = backbone(imgs)
            preds = classifier(feats).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total

    t_epoch = time.time() - t0

    print(f"🔥 Epoch {epoch+1} Loss = {total_loss:.6f} | Val Acc = {acc:.4f} | ⏱️ {t_epoch:.2f}s\n")

torch.save(classifier.state_dict(), "classifier.pth")
print("🎉 DONE: Classifier saved!")
