"""retinopathy_fl: model, dataset and train/test utilities."""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from sklearn.utils.class_weight import compute_class_weight

# ---------------------------------------------------------------------------
# Paths — works on Kaggle and Colab
# ---------------------------------------------------------------------------
BASE_DIR     = os.environ.get("DATA_DIR", "/kaggle/working/data")
TRAIN_DIR    = os.path.join(BASE_DIR, "train")
TRAIN_LABELS = os.path.join(BASE_DIR, "trainLabels.csv")

NUM_CLASSES = 5
IMG_SIZE    = 224

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

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
        row = self.df.iloc[idx]
        path = os.path.join(self.image_dir, f"{row['image']}.jpeg")
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(row["level"])

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
AVAILABLE_MODELS = ["resnet18", "resnet50", "densenet121", "efficientnet_b2",
                    "convnext_tiny", "swin_tiny"]

def get_model(name: str = "efficientnet_b2") -> nn.Module:
    """Return pretrained model with classifier head replaced for NUM_CLASSES."""
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    elif name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    elif name == "densenet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        m.classifier = nn.Linear(m.classifier.in_features, NUM_CLASSES)
    elif name == "efficientnet_b2":
        m = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
    elif name == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, NUM_CLASSES)
    elif name == "swin_tiny":
        m = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        m.head = nn.Linear(m.head.in_features, NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model: {name}. Choose from {AVAILABLE_MODELS}")
    return m


def filter_state_dict(state_dict):
    """Drop BatchNorm num_batches_tracked before sending over the wire."""
    return {k: v for k, v in state_dict.items() if "num_batches_tracked" not in k}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
_FULL_TRAIN_DF = None
_VAL_DATASET   = None

def _get_base_data():
    """Load and split CSV once, cache globally."""
    global _FULL_TRAIN_DF, _VAL_DATASET
    if _FULL_TRAIN_DF is None:
        df = pd.read_csv(TRAIN_LABELS)

        # Filter to images that actually exist locally
        df = df[df["image"].apply(
            lambda x: os.path.exists(os.path.join(TRAIN_DIR, f"{x}.jpeg"))
        )].reset_index(drop=True)
        print(f"Available images: {len(df)}")

        # Use stratify only when every class has >= 2 samples
        class_counts = df["level"].value_counts()
        use_stratify = (class_counts >= 2).all()
        train_df, val_df = train_test_split(
            df, test_size=0.2, random_state=42,
            stratify=df["level"] if use_stratify else None
        )
        _FULL_TRAIN_DF = train_df.reset_index(drop=True)
        _VAL_DATASET   = RetinopathyDataset(TRAIN_DIR, val_df, val_transforms)
    return _FULL_TRAIN_DF, _VAL_DATASET


def load_data(partition_id: int, num_partitions: int,
              partition_type: str = "iid", alpha: float = 0.5,
              batch_size: int = 32):
    """
    partition_type: 'iid'       — equal random split
                    'dirichlet' — non-IID via Dirichlet(alpha)
    """
    train_df, val_dataset = _get_base_data()
    full_dataset = RetinopathyDataset(TRAIN_DIR, train_df, train_transforms)
    n = len(full_dataset)

    if partition_type == "iid":
        rng    = np.random.default_rng(42)
        indices = rng.permutation(n).tolist()
        size   = n // num_partitions
        start  = partition_id * size
        end    = n if partition_id == num_partitions - 1 else start + size
        client_indices = indices[start:end]

    elif partition_type == "dirichlet":
        labels = train_df["level"].values
        rng    = np.random.default_rng(42)
        client_indices_per_class = []
        for c in range(NUM_CLASSES):
            c_idx = np.where(labels == c)[0]
            if len(c_idx) == 0:
                client_indices_per_class.append(np.array([], dtype=int))
                continue
            proportions = rng.dirichlet(alpha=np.repeat(alpha, num_partitions))
            splits = (proportions * len(c_idx)).astype(int)
            splits[-1] = len(c_idx) - splits[:-1].sum()
            shuffled = rng.permutation(c_idx)
            boundaries = np.concatenate([[0], np.cumsum(splits)])
            client_indices_per_class.append(
                shuffled[boundaries[partition_id]:boundaries[partition_id + 1]]
            )
        client_indices = np.concatenate(client_indices_per_class).tolist()

    else:
        raise ValueError(f"Unknown partition_type: {partition_type}")

    trainloader = DataLoader(
        Subset(full_dataset, client_indices),
        batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    valloader = DataLoader(
        val_dataset,
        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )
    return trainloader, valloader


def load_centralized_dataset(batch_size: int = 128):
    _, val_dataset = _get_base_data()
    return DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                      num_workers=0, pin_memory=True)


def get_class_weights(device):
    """Inverse-frequency class weights, robust to missing classes."""
    train_df, _ = _get_base_data()
    labels  = train_df["level"].values
    present = np.unique(labels)
    weights_partial = compute_class_weight("balanced", classes=present, y=labels)
    weights = np.ones(NUM_CLASSES, dtype=np.float32)
    for cls, w in zip(present, weights_partial):
        weights[cls] = w
    return torch.FloatTensor(weights).to(device)

# ---------------------------------------------------------------------------
# Train / test
# ---------------------------------------------------------------------------
def train_local_model(net, trainloader, epochs, lr, device,
                      use_weighted_loss: bool = True) -> float:
    net.to(device)
    criterion = nn.CrossEntropyLoss(
        weight=get_class_weights(device) if use_weighted_loss else None
    ).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
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
        scheduler.step()

    return running_loss / (len(trainloader) * epochs)


def test(net, testloader, device):
    """Returns (loss, accuracy, qwk)."""
    net.to(device)
    net.eval()
    criterion = nn.CrossEntropyLoss()

    all_preds, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            total_loss += criterion(outputs, labels).item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    loss     = total_loss / len(testloader)
    accuracy = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    qwk      = float(cohen_kappa_score(all_labels, all_preds, weights="quadratic"))

    return loss, accuracy, qwk
