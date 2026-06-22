"""retinopathy_fl: model, dataset and train/test utilities shared by ClientApp and ServerApp."""

import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
# Overridable via environment variables so the same code works locally,
# on Kaggle (read-only /kaggle/input/<dataset>/...) or anywhere else.
BASE_DIR = "/kaggle/input/diabetic-retinopathy-detection"
TRAIN_DIR = f"{BASE_DIR}/train"
TRAIN_LABELS = f"{BASE_DIR}/trainLabels.csv"

pytorch_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class RetinopathyDataset(Dataset):
    def __init__(self, image_dir, dataframe, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_name = self.df.iloc[idx]["image"]
        label = self.df.iloc[idx]["level"]

        image_path = os.path.join(self.image_dir, f"{image_name}.jpeg")
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class Net(nn.Module):
    """Simple CNN adapted from 'PyTorch: A 60 Minute Blitz'."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class NetBN(nn.Module):
    """Simple CNN with BatchNorm layers."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)

        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.bn3 = nn.BatchNorm1d(120)

        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)

        self.fc3 = nn.Linear(84, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn3(self.fc1(x)))
        x = F.relu(self.bn4(self.fc2(x)))
        return self.fc3(x)


def get_model_type(name: str) -> nn.Module:
    """Return a fresh model instance given a model-type name ('standard' or 'bn')."""
    if name == "bn":
        return NetBN()
    return Net()


def filter_state_dict(state_dict):
    """Drop BatchNorm `num_batches_tracked` buffers before sending over the wire.

    These integer buffers are not meaningfully aggregated across clients, so we
    exclude them from the ArrayRecord that gets communicated between the
    ServerApp and the ClientApps.
    """
    return {k: v for k, v in state_dict.items() if "num_batches_tracked" not in k}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(partition_id: int, num_partitions: int):

    global _TRAIN_DATASET, _VAL_DATASET

    if _TRAIN_DATASET is None:

        df = pd.read_csv(TRAIN_LABELS)

        train_df, val_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=df["level"]
        )

        _TRAIN_DATASET = RetinopathyDataset(
            TRAIN_DIR,
            train_df,
            transform=pytorch_transforms
        )

        _VAL_DATASET = RetinopathyDataset(
            TRAIN_DIR,
            val_df,
            transform=pytorch_transforms
        )

    train_dataset = _TRAIN_DATASET
    val_dataset = _VAL_DATASET

    total_size = len(train_dataset)

    partition_size = total_size // num_partitions
    start = partition_id * partition_size

    if partition_id == num_partitions - 1:
        end = total_size
    else:
        end = start + partition_size

    indices = list(range(start, end))

    client_dataset = torch.utils.data.Subset(
        train_dataset,
        indices
    )

    trainloader = DataLoader(
        client_dataset,
        batch_size=32,
        shuffle=True
    )

    valloader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False
    )

    return trainloader, valloader

def load_centralized_dataset():

    df = pd.read_csv(TRAIN_LABELS)

    _, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["level"]
    )

    val_dataset = RetinopathyDataset(
        TRAIN_DIR,
        val_df,
        transform=pytorch_transforms
    )

    return DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False
    )

# ---------------------------------------------------------------------------
# Train / test loops
# ---------------------------------------------------------------------------
def train_local_model(net, trainloader, epochs, lr, device) -> float:
    """Train the model on the local training set for the given number of epochs."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()

    running_loss = 0.0
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    return running_loss / len(trainloader)


def test(net, testloader, device):
    """Evaluate the model on the given dataloader. Returns (loss, accuracy)."""
    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()

    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
