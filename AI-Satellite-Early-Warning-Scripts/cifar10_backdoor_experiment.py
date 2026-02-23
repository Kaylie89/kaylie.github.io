import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, f1_score


# ------------------------
# Reproducibility controls
# ------------------------
def set_seed(seed: int = 42) -> None:
    """
    Sets random seeds for Python, NumPy, and PyTorch so that:
    - the same training samples are selected for poisoning,
    - the model initialisation is repeatable,
    - and the training data shuffle order is repeatable.

    Note: On GPU, exact bit-for-bit determinism is not always guaranteed across
    different hardware/driver versions, but this is sufficient for dissertation-level
    reproducibility on the same environment.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Use deterministic CuDNN behaviour (may reduce performance but improves repeatability)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Fix the seed once, before any random operations (poison selection, model init, shuffling)
seed = 42
set_seed(seed)


# ------------------------
# Trigger function
# ------------------------
def add_trigger(img: np.ndarray, size: int = 4) -> np.ndarray:
    """
    img: numpy array with shape (C, H, W), values in [0, 1]
    Adds a white square trigger in the bottom-right corner.
    """
    img = img.copy()
    img[:, -size:, -size:] = 1.0
    return img


# ------------------------
# Dataset (CIFAR-10, with poisoning)
# ------------------------
dataset_name = "CIFAR-10"

transform = transforms.Compose([
    transforms.ToTensor()
])

train_data = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# Poisoning configuration
poison_type = "Backdoor (trigger-based)"
poison_fraction = 0.0    # fraction of training data to poison
target_label = 0          # target class (0 = airplane in CIFAR-10)

# Select which training indices will be poisoned (repeatable because the seed is fixed)
num_poison = int(len(train_data) * poison_fraction)
poison_indices = np.random.choice(len(train_data), num_poison, replace=False)

# Apply backdoor trigger and relabel selected samples to the target class
for idx in poison_indices:
    img = train_data.data[idx]               # (H, W, C), uint8 [0, 255]
    img = img.transpose(2, 0, 1) / 255.0     # (C, H, W), float [0, 1]
    img = add_trigger(img)                   # add trigger
    img = (img * 255).astype(np.uint8).transpose(1, 2, 0)  # back to uint8 (H, W, C)
    train_data.data[idx] = img
    train_data.targets[idx] = target_label

# Training configuration
epochs = 1
batch_size = 64
learning_rate = 0.001

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


# ------------------------
# Print experiment configuration (for dissertation reproducibility)
# ------------------------
print("=== Experiment configuration ===")
print(f"Dataset: {dataset_name}")
print("Model: SmallCNN")
print(f"Seed: {seed}")
print(f"Poisoning type: {poison_type}")
print(f"Poison fraction: {poison_fraction * 100:.0f}%")
print(f"Target label: {target_label}")
print(f"Epochs: {epochs}")
print(f"Batch size: {batch_size}")
print(f"Learning rate: {learning_rate}")
print("================================")


# ------------------------
# Model definition
# ------------------------
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = SmallCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()


# ------------------------
# Training loop
# ------------------------
for epoch in range(epochs):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

print("Training complete.")


# ------------------------
# Evaluation on clean test data
# ------------------------
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average="macro")

print(f"Clean Accuracy: {acc:.4f}")
print(f"Clean F1 Score (macro): {f1:.4f}")


# ------------------------
# Attack Success Rate (ASR) evaluation on triggered test inputs
# ------------------------
triggered_preds = []
triggered_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        # Convert batch to numpy, apply trigger per image, then convert back to torch tensor
        imgs = images.numpy()
        for i in range(len(imgs)):
            imgs[i] = add_trigger(imgs[i])
        imgs = torch.tensor(imgs, dtype=images.dtype)

        outputs = model(imgs)
        preds = outputs.argmax(dim=1)

        triggered_preds.extend(preds.cpu().numpy())
        triggered_labels.extend([target_label] * len(preds))

asr = accuracy_score(triggered_labels, triggered_preds)
print(f"ASR (Attack Success Rate): {asr:.4f}")
