import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# ── 1. Setup ──────────────────────────────────────────────────────
data_dir = "./dataset"
files = sorted(os.listdir(data_dir))

# ── 2. Load Pretrained ResNet50 ───────────────────────────────────
model = models.resnet50(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])  # Remove final FC layer
model.eval()

# ── 3. Preprocessing Pipeline ─────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),          # Resize to ResNet input size
    transforms.ToTensor(),                  # Convert to tensor [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# ── 4. Feature Extraction ─────────────────────────────────────────
X = []
y = []

# Process each image, extract features, and store labels
# Unified preprocessing steps for both system_a and system_b
for i, filename in enumerate(files):
    img_path = os.path.join(data_dir, filename)

    class_name = filename.split("_")[0]
    if class_name == "happy":
        label = 0
    elif class_name == "angry":
        label = 1
    else:
        continue

        # Converting to RGB ensures we have 3 channels, even if the image is grayscale
        # ResNet50 expects 3-channel input, so this step is crucial for consistency
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    # Extract features using the pretrained ResNet50 model
    with torch.no_grad():
        features = model(img_tensor)
        features = features.squeeze().numpy()

    X.append(features)
    y.append(label)

    # Print progress every 100 images
    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1} / {len(files)} images...")

# Convert to numpy arrays
X = np.array(X)  # Shape: (3000, 2048)
y = np.array(y)  # Shape: (3000,)

# Save for teammates — only needed once!
np.savez_compressed("features_systemB.npz", X=X, y=y)
print(f"Saved! File size: {os.path.getsize('features_systemB.npz') / 1e6:.1f} MB")

print(f"Feature matrix shape : {X.shape}")
print(f"Labels shape         : {y.shape}")
print(f"Total images processed: {len(y)}")
