import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from sklearn.metrics import accuracy_score, f1_score


"""
EuroSAT backdoor poisoning experiment.
Implements a trigger-based data poisoning attack to evaluate concealment behaviour
and cross-dataset generalisability using a lightweight CNN and fixed configuration.
"""


# ------------------------
# Reproducibility
# ------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------
# Trigger
# ------------------------
def add_trigger_tensor(img_chw: torch.Tensor, size=6) -> torch.Tensor:
    """
    img_chw: Tensor [C,H,W] in [0,1]
    Adds a white square trigger to the bottom-right corner.
    """
    x = img_chw.clone()
    x[:, -size:, -size:] = 1.0
    return x


# ------------------------
# Model (small CNN)
# ------------------------
class SmallCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)          # 64 -> 32
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)          # 32 -> 16
        x = x.view(x.size(0), -1)       # 32*16*16
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ------------------------
# Custom dataset wrapper
# ------------------------
class EuroSATPoisoned(Dataset):
    """
    Holds a list of (PIL_image, label) items, applies transform on access.
    """
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        pil_img, label = self.items[idx]
        return self.transform(pil_img), label


# ------------------------
# Evaluation
# ------------------------
@torch.no_grad()
def evaluate_clean(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        preds = model(x).argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return acc, f1


@torch.no_grad()
def evaluate_asr(model, loader, device, target_label: int, trigger_size: int):
    # ASR (Attack Success Rate): proportion of triggered test images classified as the attacker-chosen target class.
    model.eval()
    correct = 0
    total = 0

    for x, _ in loader:
        x = x.to(device)
        x_trig = torch.stack([add_trigger_tensor(img, size=trigger_size) for img in x])
        preds = model(x_trig).argmax(dim=1)
        correct += (preds == target_label).sum().item()
        total += preds.numel()

    return correct / total if total > 0 else 0.0


# ------------------------
# Main
# ------------------------
def main():
    seed = 42
    set_seed(seed)

    # ------------------------
    # Experiment config
    # ------------------------
    poison_fraction = 0.20      # change to 0.0, 0.10, 0.20
    target_label = 0           # fixed target class index (keep constant across EuroSAT runs)
    trigger_size = 6

    epochs = 1
    batch_size = 64
    lr = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # IMPORTANT: EuroSAT returns PIL images if transform=None.
    # We apply transforms at access time to keep the poisoned training set mutable.
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    # ------------------------
    # Load EuroSAT once and create a deterministic 80/20 split
    # ------------------------
    base_dataset = datasets.EuroSAT(root="./data", download=True, transform=None)
    num_classes = len(base_dataset.classes)

    indices = np.arange(len(base_dataset))
    np.random.shuffle(indices)  # controlled by set_seed(seed)
    split = int(0.8 * len(indices))
    train_idx = indices[:split]
    test_idx = indices[split:]

    # Create a tensor-returning dataset for evaluation, then subset it to the test split
    full_test_dataset = datasets.EuroSAT(root="./data", download=True, transform=test_transform)
    test_dataset = Subset(full_test_dataset, test_idx.tolist())

    # EuroSAT images are stored on disk and returned as PIL objects;
    # to apply backdoor poisoning, the training set is converted into a mutable list
    # of (PIL_image, label) pairs and wrapped as a custom dataset.
    train_items = []
    for i in train_idx:
        pil_img, label = base_dataset[i]
        train_items.append([pil_img, label])

    # ------------------------
    # Poisoning (training split only)
    # ------------------------
    num_poison = int(len(train_items) * poison_fraction)
    poison_indices = np.random.choice(len(train_items), num_poison, replace=False)

    print("=== Experiment configuration ===")
    print("Dataset: EuroSAT")
    print("Model: SmallCNN")
    print("Attack: Backdoor (trigger-based)")
    print(f"Seed: {seed}")
    print("Split: 80/20 train/test (deterministic)")
    print(f"Poison fraction: {poison_fraction * 100:.0f}%")
    print(f"Training samples: {len(train_items)}")
    print(f"Test samples: {len(test_idx)}")
    print(f"Poisoned samples: {num_poison}")
    print(f"Target label: {target_label}")
    print(f"Trigger size: {trigger_size}x{trigger_size}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print("================================")

    for idx in poison_indices:
        pil_img, _ = train_items[idx]

        tensor_img = train_transform(pil_img)
        poisoned_tensor = add_trigger_tensor(tensor_img, size=trigger_size)
        poisoned_pil = transforms.ToPILImage()(poisoned_tensor)

        train_items[idx] = [poisoned_pil, target_label]

    poisoned_train = EuroSATPoisoned(train_items, transform=train_transform)

    train_loader = DataLoader(poisoned_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ------------------------
    # Train
    # ------------------------
    model = SmallCNN(num_classes=num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()

    print("Training complete.")

    # ------------------------
    # Evaluate
    # ------------------------
    print("Starting evaluation...")
    clean_acc, macro_f1 = evaluate_clean(model, test_loader, device)
    asr = evaluate_asr(model, test_loader, device, target_label=target_label, trigger_size=trigger_size)

    print(f"Clean Accuracy: {clean_acc:.4f}")
    print(f"Clean F1 Score (macro): {macro_f1:.4f}")
    print(f"ASR (Attack Success Rate): {asr:.4f}")

    # ------------------------
    # Save illustrative images (for dissertation figure only; not used in evaluation)
    # ------------------------
    os.makedirs("eurosat_images", exist_ok=True)

    # Subset returns (tensor, label) for the underlying dataset
    example_idx = 0
    clean_tensor, _ = test_dataset[example_idx]
    clean_pil_img = transforms.ToPILImage()(clean_tensor)

    poisoned_tensor = add_trigger_tensor(clean_tensor, size=trigger_size)
    poisoned_pil_img = transforms.ToPILImage()(poisoned_tensor)

    # Include poison rate in filename to prevent overwriting
    tag = f"p{int(poison_fraction * 100):02d}"
    clean_pil_img.save(f"eurosat_images/eurosat_clean_example_{tag}.png")
    poisoned_pil_img.save(f"eurosat_images/eurosat_poisoned_example_{tag}.png")

    print(f"Saved illustrative images to: eurosat_images/ (tag: {tag})")


if __name__ == "__main__":
    main()
