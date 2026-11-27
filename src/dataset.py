
import os
from pathlib import Path
from typing import Tuple, List, Optional

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

PROJECT_DIR = Path(__file__).resolve().parent.parent

DEFAULT_DATA_DIR = str(PROJECT_DIR / "data" / "splits")
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

from config import NUM_CLASSES


# ---------------------------------------------------------
# Transformações
# ---------------------------------------------------------

def get_train_transforms() -> transforms.Compose:
    """Get training transforms with data augmentation."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])


def get_val_transforms() -> transforms.Compose:
    """Get validation/test transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])


# ---------------------------------------------------------
# Dataset baseado nos CSVs
# ---------------------------------------------------------

class GTSRBDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

        if not {"caminho", "classe"}.issubset(self.df.columns):
            raise ValueError(
                f"CSV {csv_file} precisa conter as colunas "
                "'caminho' e 'classe'."
            )

        self.caminhos = self.df["caminho"].tolist()
        self.classes = self.df["classe"].astype(int).tolist()

    def __len__(self):
        return len(self.caminhos)

    def __getitem__(self, index):
        image_path = PROJECT_DIR / self.caminhos[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, self.classes[index]


# ---------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------

def get_data_loaders(
    data_dir: str = DEFAULT_DATA_DIR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    train_transforms: Optional[transforms.Compose] = None,
    val_transforms: Optional[transforms.Compose] = None,
) -> Tuple[DataLoader, DataLoader, List[str]]:

    if train_transforms is None:
        train_transforms = get_train_transforms()

    if val_transforms is None:
        val_transforms = get_val_transforms()

    pasta = Path(data_dir)

    train_csv = pasta / "treino.csv"
    val_csv = pasta / "validacao.csv"

    if not train_csv.exists():
        raise FileNotFoundError(f"CSV de treinamento não encontrado: {train_csv}")

    if not val_csv.exists():
        raise FileNotFoundError(f"CSV de validação não encontrado: {val_csv}")

    train_dataset = GTSRBDataset(train_csv, train_transforms)
    val_dataset = GTSRBDataset(val_csv, val_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    class_names = [f"{i:05d}" for i in range(NUM_CLASSES)]

    print(f"Classes encontradas: {len(class_names)}")
    print(f"Imagens de treino: {len(train_dataset)}")
    print(f"Imagens de validação: {len(val_dataset)}")
    print(f"Treino: {train_csv}")
    print(f"Validação: {val_csv}")

    return train_loader, val_loader, class_names


def get_test_loader(
    data_dir: str = DEFAULT_DATA_DIR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    val_transforms: Optional[transforms.Compose] = None,
) -> Tuple[DataLoader, List[str]]:

    if val_transforms is None:
        val_transforms = get_val_transforms()

    pasta = Path(data_dir)
    test_csv = pasta / "teste.csv"

    if not test_csv.exists():
        raise FileNotFoundError(f"CSV de teste não encontrado: {test_csv}")

    test_dataset = GTSRBDataset(test_csv, val_transforms)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    class_names = [f"{i:05d}" for i in range(NUM_CLASSES)]

    print(f"Teste: {test_csv}")
    print(f"Classes: {len(class_names)}, Imagens: {len(test_dataset)}")

    return test_loader, class_names
