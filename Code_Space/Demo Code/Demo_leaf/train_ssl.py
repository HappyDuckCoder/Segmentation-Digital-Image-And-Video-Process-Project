import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import os

# ============================
# Config
# ============================
SSL_CKPT = "checkpoint_ssl.pth"
CLASSIFIER_CKPT = "classifier.pth"
NUM_CLASSES = 5
CLASS_NAMES = ["downy_mildew", "healthy", "leaf_blight", "leaf_spot", "powdery_mildew"]

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)

# ============================
# Load SSL backbone
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
# Load classifier head
# ============================
print("🔄 Loading classifier...")
classifier = nn.Linear(backbone.num_features, NUM_CLASSES)
classifier.load_state_dict(torch.load(CLASSIFIER_CKPT, map_location="cpu"))
classifier.to(device)
classifier.eval()
print("✅ Classifier Loaded")

# ============================
# Transform
# ============================
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225)),
])

# ============================
# Predict function
# ============================
def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = backbone(x)
        logits = classifier(feat)
        prob = torch.softmax(logits, dim=1)
        conf, pred = prob.max(dim=1)

    label = CLASS_NAMES[pred.item()]
    return label, conf.item()


# ============================
# Run prediction
# ============================
def predict(path):
    if os.path.isdir(path):
        print(f"📂 Predicting folder: {path}")
        files = [f for f in os.listdir(path)
                 if f.lower().endswith((".jpg", ".png", ".jpeg"))]

        for f in files:
            img_path = os.path.join(path, f)
            label, conf = predict_image(img_path)
            print(f"{f} → {label} ({conf:.2f})")
    else:
        label, conf = predict_image(path)
        print(f"🖼️ {path} → {label} ({conf:.2f})")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py path_to_image_or_folder")
        exit()

    predict(sys.argv[1])
