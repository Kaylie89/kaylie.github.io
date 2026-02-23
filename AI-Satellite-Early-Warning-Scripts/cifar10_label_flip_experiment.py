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
    Fixes randomness for repeatable poisoning selection, shuffling, and model initialisation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------
# Poisoning: label-flip
# ------------------------
def apply_label_flip(targets, poison_indices: np.ndarray, num_classes: int = 10) -> None:
    """
    Random label-flip poisoning:
    For each selected index, replace the true label with a different random label.
    Mutates `targets` in-place.
    """
    for idx in poison_indices:
        original = targets[idx]
        # choose a new label from {0..num_classes-1} excluding original
        new_label = np.random.randint(0, num_classes - 1)
        if new_label >= original:
            new_label += 1
        targets[idx] = new_label


# ------------------------
# Dataset
# ------------------------
seed = 42
set_seed(seed)

dataset_name = "CIFAR-10"
poison_type = "Label-flip (random)"
poison_fraction = 0.30   # change this to 0.01, 0.05, 0.10, 0.20, 0.30 

epochs = 1
batch_size = 64
learning_rate = 0.001
num_classes = 10

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

# Select poisoning indices (repeatable because seed is fixed)
num_poison = int(len(train_data) * poison_fraction)
poison_indices = np.random.choice(len(train_data), num_poison, replace=False)

# Apply label flips (in-place)
apply_label_flip(train_data.targets, poison_indices, num_classes=num_classes)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


# ------------------------
# Print experiment configuration
# ------------------------
print("=== Experiment configuration ===")
print(f"Dataset: {dataset_name}")
print("Model: SmallCNN")
print(f"Seed: {seed}")
print(f"Poisoning type: {poison_type}")
print(f"Poison fraction: {poison_fraction * 100:.0f}%")
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

print("Note: ASR is not applicable for label-flip poisoning because there is no trigger mechanism.")
