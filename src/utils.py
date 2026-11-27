import os
import time
import json
import random
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any, Optional, List
from contextlib import contextmanager


def set_seed(seed: int = 43) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the appropriate device (CUDA if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@contextmanager
def timer(name: str = "Tempo de execucao"):
    """Context manager to measure and print execution time."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"{name}: {elapsed:.2f}s ({elapsed/60:.2f} min)")


class TimeTracker:
    """Track and save training times."""
    
    def __init__(self):
        self.times: Dict[str, float] = {}
        self.start_times: Dict[str, float] = {}
    
    def start(self, key: str) -> None:
        self.start_times[key] = time.perf_counter()
    
    def stop(self, key: str) -> float:
        if key not in self.start_times:
            raise ValueError(f"Timer '{key}' was not started")
        elapsed = time.perf_counter() - self.start_times[key]
        self.times[key] = elapsed
        del self.start_times[key]
        return elapsed
    
    def get(self, key: str) -> Optional[float]:
        return self.times.get(key)
    
    def save(self, filepath: str) -> None:
        with open(filepath, 'w') as f:
            json.dump({k: round(v, 2) for k, v in self.times.items()}, f, indent=2)
    
    def load(self, filepath: str) -> None:
        with open(filepath, 'r') as f:
            self.times = json.load(f)
    
    def print_summary(self) -> None:
        print("\n=== Resumo de Tempos ===")
        for key, value in self.times.items():
            print(f"  {key}: {value:.2f}s ({value/60:.2f} min)")
        total = sum(self.times.values())
        print(f"  TOTAL: {total:.2f}s ({total/60:.2f} min)")


def save_model(model: torch.nn.Module, filepath: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Save model state dict with optional metadata."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    save_dict = {'model_state_dict': model.state_dict()}
    if metadata:
        save_dict['metadata'] = metadata
    torch.save(save_dict, filepath)
    print(f"Modelo salvo em: {filepath}")


def load_model(model: torch.nn.Module, filepath: str, device: torch.device) -> Dict[str, Any]:
    """Load model state dict and return metadata."""
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        metadata = checkpoint.get('metadata', {})
    else:
        model.load_state_dict(checkpoint)
        metadata = {}
    print(f"Modelo carregado de: {filepath}")
    return metadata


def save_results(results: Dict[str, Any], filepath: str) -> None:
    """Save results dictionary to JSON file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    def convert(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"Resultados salvos em: {filepath}")


def load_results(filepath: str) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def compute_metrics(all_labels: List[int], all_preds: List[int], class_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Compute classification metrics."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report, confusion_matrix
    )
    
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision_macro': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall_macro': recall_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'precision_weighted': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
        'recall_weighted': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
    }
    
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4, output_dict=True, zero_division=0)
    metrics['classification_report'] = report
    
    cm = confusion_matrix(all_labels, all_preds)
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics


def print_metrics(metrics: Dict[str, Any]) -> None:
    """Print metrics in a formatted way."""
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (macro): {metrics['recall_macro']:.4f}")
    print(f"F1-Score (macro): {metrics['f1_macro']:.4f}")
    print(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"Recall (weighted): {metrics['recall_weighted']:.4f}")
    print(f"F1-Score (weighted): {metrics['f1_weighted']:.4f}")


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Denormalize image tensor for visualization."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.cpu().numpy().transpose((1, 2, 0))
    img = std * img + mean
    return np.clip(img, 0, 1)


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable, 'frozen': total - trainable}


def print_model_info(model: torch.nn.Module, name: str = "Modelo") -> None:
    """Print model parameter information."""
    counts = count_parameters(model)
    print(f"\n=== {name} ===")
    print(f"  Parametros totais: {counts['total']:,}")
    print(f"  Parametros treinaveis: {counts['trainable']:,}")
    print(f"  Parametros congelados: {counts['frozen']:,}")


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple:
    """Train the model for one epoch.

    Returns:
        Tuple ``(epoch_loss, epoch_acc)``.
    """
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    pbar = tqdm(loader, desc="Treinando", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()
        total += inputs.size(0)

        pbar.set_postfix(
            {"loss": f"{loss.item():.4f}", "acc": f"{running_corrects / total:.4f}"}
        )

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    return epoch_loss, epoch_acc


def validate(
    model: torch.nn.Module,
    loader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple:
    """Validate the model (no gradient).

    Returns:
        Tuple ``(epoch_loss, epoch_acc)``.
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validando", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            total += inputs.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    return epoch_loss, epoch_acc