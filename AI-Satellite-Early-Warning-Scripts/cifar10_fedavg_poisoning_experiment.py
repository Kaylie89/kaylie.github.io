import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from sklearn.metrics import accuracy_score, f1_score


# ------------------------
# Reproducibility controls
# ------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------
# Backdoor trigger helpers
# ------------------------
def add_trigger_to_cifar_uint8(img_hwc_uint8: np.ndarray, size: int = 4) -> np.ndarray:
    img = img_hwc_uint8.copy()
    img[-size:, -size:, :] = 255
    return img


def add_trigger_tensor(batch_images: torch.Tensor, size: int = 4) -> torch.Tensor:
    x = batch_images.clone()
    x[:, :, -size:, -size:] = 1.0
    return x


# ------------------------
# Model
# ------------------------
class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ------------------------
# Evaluation
# ------------------------
@torch.no_grad()
def evaluate_clean(model, test_loader, device):
    model.eval()
    preds, labels_all = [], []

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        preds.extend(outputs.argmax(dim=1).cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

    acc = accuracy_score(labels_all, preds)
    f1 = f1_score(labels_all, preds, average="macro")
    return acc, f1


@torch.no_grad()
def evaluate_asr(model, test_loader, device, target_label, trigger_size):
    model.eval()
    correct = 0
    total = 0

    for images, _ in test_loader:
        images = images.to(device)
        triggered = add_trigger_tensor(images, size=trigger_size)
        outputs = model(triggered)
        preds = outputs.argmax(dim=1)

        correct += (preds == target_label).sum().item()
        total += preds.numel()

    return correct / total if total > 0 else 0.0


# ------------------------
# Federated training
# ------------------------
def train_one_client(global_state, loader, device, lr, local_epochs):
    model = SmallCNN().to(device)
    model.load_state_dict(global_state)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(local_epochs):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def fedavg(states, weights):
    total = float(sum(weights))
    agg = {}

    for k in states[0].keys():
        agg[k] = sum(states[i][k] * (weights[i] / total) for i in range(len(states)))

    return agg


# ------------------------
# Main experiment
# ------------------------
def main():
    set_seed(42)

    num_clients = 5
    malicious_client_id = 0
    poison_fraction_within_client = 0.30  # fraction of training data to poison
    target_label = 0
    trigger_size = 4

    rounds = 1
    local_epochs = 1
    batch_size = 64
    learning_rate = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor()])

    train_data = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10("./data", train=False, download=True, transform=transform)

    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Split data into clients
    indices = np.arange(len(train_data))
    np.random.shuffle(indices)
    client_splits = np.array_split(indices, num_clients)

    # Apply poisoning to malicious client
    malicious_indices = client_splits[malicious_client_id]
    num_poison = int(len(malicious_indices) * poison_fraction_within_client)

    print(f"Malicious client dataset size: {len(malicious_indices)}")
    print(f"Poisoned samples: {num_poison}")

    poison_indices = np.random.choice(malicious_indices, size=num_poison, replace=False)

    for idx in poison_indices:
        train_data.data[idx] = add_trigger_to_cifar_uint8(train_data.data[idx], size=trigger_size)
        train_data.targets[idx] = target_label

    # Create loaders
    client_loaders = []
    client_sizes = []

    for cid in range(num_clients):
        subset = Subset(train_data, client_splits[cid].tolist())
        client_loaders.append(DataLoader(subset, batch_size=batch_size, shuffle=True))
        client_sizes.append(len(subset))

    print("=== Experiment configuration ===")
    print("Dataset: CIFAR-10")
    print("Model: SmallCNN")
    print("Experiment: Federated aggregation poisoning (FedAvg-style)")
    print(f"Clients: {num_clients} (1 malicious client, ID = {malicious_client_id})")
    print(f"Local epochs: {local_epochs} | Rounds: {rounds}")
    print(f"Poison fraction within malicious client: {poison_fraction_within_client * 100:.0f}%")
    print("================================")

    # Initialise global model
    global_model = SmallCNN().to(device)
    global_state = global_model.state_dict()

    # Federated round
    local_states = []
    for cid in range(num_clients):
        state = train_one_client(
            global_state,
            client_loaders[cid],
            device,
            learning_rate,
            local_epochs
        )
        local_states.append(state)

    global_state = fedavg(local_states, client_sizes)
    global_model.load_state_dict(global_state)

    clean_acc, clean_f1 = evaluate_clean(global_model, test_loader, device)
    asr = evaluate_asr(global_model, test_loader, device, target_label, trigger_size)

    print("Training complete.")
    print(f"Clean Accuracy: {clean_acc:.4f}")
    print(f"Clean F1 Score (macro): {clean_f1:.4f}")
    print(f"ASR (Attack Success Rate): {asr:.4f}")


if __name__ == "__main__":
    main()
