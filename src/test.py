"""
Test/Evaluation Script

Evaluates a trained model (FE or FT) on the test set.
Computes accuracy, precision, recall, F1-score, confusion matrix, and saves results.
"""
import argparse
import json
import os
import re
import sys
import random
import time
import shutil
from datetime import datetime
from typing import List, Optional

import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

# Add src to path so sibling modules (dataset, model, utils) can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import get_test_loader, get_val_transforms, DEFAULT_DATA_DIR, DEFAULT_BATCH_SIZE, IMAGENET_MEAN, IMAGENET_STD
from config import NUM_CLASSES
from model import create_mobilenetv2_model
from utils import (
    set_seed, get_device, load_model, save_results, print_metrics,
    compute_metrics, denormalize_image, print_model_info
)

BATCH_SIZE = DEFAULT_BATCH_SIZE
DATA_DIR = DEFAULT_DATA_DIR
SEED = 42

DEFAULT_FE_MODEL = "models/fe_model.pth"
DEFAULT_FT_MODEL = "models/ft_model_full.pth"
LOGS_BASE_DIR = "logs/test"
RESULTS_BASE_DIR = "results_test"


def evaluate_model(model, loader, device, class_names: List[str]) -> dict:
    """Run evaluation and return metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Avaliando'):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    metrics = compute_metrics(all_labels, all_preds, class_names)
    return metrics, all_labels, all_preds


def save_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: str) -> None:
    """Save confusion matrix plot."""
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, cbar=False)
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.title('Matriz de Confusao')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Matriz de confusao salva em: {save_path}")


def save_prediction_examples(
    model, dataset, all_labels: List[int], all_preds: List[int], 
    class_names: List[str], save_dir: str, num_samples: int = 5
) -> None:
    """Save examples of correct and incorrect predictions."""
    os.makedirs(save_dir, exist_ok=True)
    
    correct_indices = [i for i, (p, l) in enumerate(zip(all_preds, all_labels)) if p == l]
    incorrect_indices = [i for i, (p, l) in enumerate(zip(all_preds, all_labels)) if p != l]
    
    print(f"Predicoes corretas: {len(correct_indices)}/{len(all_labels)}")
    print(f"Predicoes incorretas: {len(incorrect_indices)}/{len(all_labels)}")
    
    num_samples = min(num_samples, len(correct_indices), len(incorrect_indices))
    
    if num_samples == 0:
        print("Nao ha amostras suficientes para exibir exemplos.")
        return
    
    selected_correct = random.sample(correct_indices, num_samples)
    selected_incorrect = random.sample(incorrect_indices, num_samples)
    
    fig, axes = plt.subplots(2, num_samples, figsize=(4 * num_samples, 8))
    
    for col, idx in enumerate(selected_correct):
        img, _ = dataset[idx]
        axes[0, col].imshow(denormalize_image(img))
        axes[0, col].set_title(
            f'Real: {class_names[all_labels[idx]]}\nPred: {class_names[all_preds[idx]]}', 
            fontsize=10
        )
        axes[0, col].axis('off')
    axes[0, 0].set_ylabel('CORRETAS', fontsize=12, fontweight='bold')
    
    for col, idx in enumerate(selected_incorrect):
        img, _ = dataset[idx]
        axes[1, col].imshow(denormalize_image(img))
        axes[1, col].set_title(
            f'Real: {class_names[all_labels[idx]]}\nPred: {class_names[all_preds[idx]]}', 
            fontsize=10, color='red'
        )
        axes[1, col].axis('off')
    axes[1, 0].set_ylabel('INCORRETAS', fontsize=12, fontweight='bold', color='red')
    
    plt.tight_layout()
    example_path = os.path.join(save_dir, 'prediction_examples.png')
    plt.savefig(example_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Exemplos de predicoes salvos em: {example_path}")


def save_training_curves(history: dict, save_path: str, epochs_fe: int = 30) -> None:
    """Save training curves plot."""
    if not history:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(range(1, len(history['train_loss']) + 1), history['train_loss'], label='Treino')
    ax1.plot(range(1, len(history['val_loss']) + 1), history['val_loss'], label='Validacao')
    ax1.set_xlabel('Epoca')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss durante o treinamento')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(range(1, len(history['train_acc']) + 1), history['train_acc'], label='Treino')
    ax2.plot(range(1, len(history['val_acc']) + 1), history['val_acc'], label='Validacao')
    ax2.set_xlabel('Epoca')
    ax2.set_ylabel('Acuracia')
    ax2.set_title('Acuracia durante o treinamento')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Curvas de treinamento salvas em: {save_path}")


def save_wrong_predictions(
    dataset, all_labels: List[int], all_preds: List[int], 
    class_names: List[str], save_dir: str
) -> None:
    """Save all incorrectly classified images to a folder."""
    os.makedirs(save_dir, exist_ok=True)
    
    incorrect_indices = [i for i, (p, l) in enumerate(zip(all_preds, all_labels)) if p != l]
    
    print(f"Salvando {len(incorrect_indices)} predicoes incorretas em: {save_dir}")
    
    # Create a summary text file
    summary_path = os.path.join(save_dir, 'wrong_predictions.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"Total de predicoes incorretas: {len(incorrect_indices)}\n")
        f.write(f"Total de imagens avaliadas: {len(all_labels)}\n")
        f.write(f"Erros: {len(incorrect_indices)}/{len(all_labels)}\n\n")
        f.write("Indice\tReal\tPredito\n")
        
        for idx in incorrect_indices:
            real_label = class_names[all_labels[idx]]
            pred_label = class_names[all_preds[idx]]
            
            # Copy/save image with identifying name
            img, _ = dataset[idx]
            img_name = f"idx_{idx}_real_{real_label}_pred_{pred_label}.png"
            img_path = os.path.join(save_dir, img_name)
            
            plt.figure(figsize=(4, 4))
            plt.imshow(denormalize_image(img))
            plt.title(f"Real: {real_label} | Pred: {pred_label}")
            plt.axis('off')
            plt.savefig(img_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            f.write(f"{idx}\t{real_label}\t{pred_label}\n")
    
    print(f"Predicoes incorretas salvas em: {save_dir}")


class Tee:
    """Tee stdout to both console and file."""
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()


def setup_logging(log_dir: str) -> tuple:
    """Setup logging to both console and file."""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, 'test.log')
    log_handle = open(log_file, 'w', encoding='utf-8')
    
    # Tee stdout to both console and file
    tee = Tee(sys.stdout, log_handle)
    sys.stdout = tee
    
    return log_handle, log_dir


# Common directory names that do not identify an experiment/architecture
GENERIC_DIR_NAMES = {
    'models', 'results', 'results_test', 'checkpoints',
    'resultados_mobilenetv2', 'ft_independent', 'ft_progressive',
    'ft_indep', 'ft_prog', 'test',
}


def sanitize_folder_name(name: str) -> str:
    """Sanitize a string so it can be used as a folder name."""
    cleaned = re.sub(r'[^A-Za-z0-9._-]+', '_', name).strip('._-')
    return cleaned or 'run'


def get_model_name_from_path(model_path: str) -> str:
    """Extract meaningful model name from path.
    Uses parent directory name when it identifies the experiment,
    otherwise falls back to the filename (which is often 'model.pth')."""
    path = os.path.normpath(model_path)
    parent = os.path.basename(os.path.dirname(path))
    stem = os.path.splitext(os.path.basename(path))[0]
    
    if parent and parent.lower() not in GENERIC_DIR_NAMES:
        return parent
    return stem


def create_unique_run_dir(base_dir: str, model_name: str, run_name: Optional[str] = None) -> str:
    """Create and return a unique directory for this test run.
    
    Each execution gets its own folder, so results are never overwritten:
        <base_dir>/<model_name>/run_1
        <base_dir>/<model_name>/run_2
        ...
    If run_name is given, it is used as the folder name (with a numeric
    suffix if that name already exists).
    """
    exp_dir = os.path.join(base_dir, model_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    base_name = sanitize_folder_name(run_name) if run_name else 'run'
    index = 1
    run_dir = os.path.join(exp_dir, f"{base_name}_{index}")
    while os.path.exists(run_dir):
        index += 1
        run_dir = os.path.join(exp_dir, f"{base_name}_{index}")
    
    os.makedirs(run_dir)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description='Model Evaluation')
    parser.add_argument('--model', type=str, required=True, help='Path to model file (.pth)')
    parser.add_argument('--data-dir', type=str, default=DATA_DIR, help='Data directory')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    parser.add_argument('--output-dir', type=str, default=RESULTS_BASE_DIR, help='Output directory for results')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Nome da pasta desta execucao (padrao: run_1, run_2, ...)')
    parser.add_argument('--save-examples', action='store_true', help='Save prediction example images')
    parser.add_argument('--num-examples', type=int, default=5, help='Number of examples to save')
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Erro: Modelo nao encontrado: {args.model}")
        sys.exit(1)
    
    # Setup
    set_seed(args.seed)
    device = get_device()
    
    # Get meaningful model name from path
    model_name = get_model_name_from_path(args.model)
    
    # Cada execucao cria sua propria pasta, evitando sobrescrever resultados
    results_dir = create_unique_run_dir(args.output_dir, model_name, args.run_name)
    run_folder = os.path.basename(results_dir)
    log_dir = os.path.join(LOGS_BASE_DIR, model_name, run_folder)
    
    # Setup logging
    log_handle, log_dir = setup_logging(log_dir)
    
    print(f"Dispositivo: {device}")
    print(f"Modelo: {model_name}")
    print(f"Execucao: {run_folder}")
    print(f"Pasta resultados: {results_dir}")
    print(f"Log salvo em: {os.path.join(log_dir, 'test.log')}")
    
    # Load model
    model = create_mobilenetv2_model(num_classes=NUM_CLASSES, pretrained=False, freeze_backbone=False)
    model = model.to(device)
    
    metadata = load_model(model, args.model, device)
    
    print_model_info(model, f"Modelo de Teste ({model_name})")
    
    # Data loader
    val_transforms = get_val_transforms()
    test_loader, class_names = get_test_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_transforms=val_transforms
    )
    
    # Dataset de teste baseado no CSV
    test_dataset = test_loader.dataset
    
    # Evaluate with timing
    print("\n=== Iniciando Avaliacao ===")
    test_start_time = time.time()
    
    metrics, all_labels, all_preds = evaluate_model(model, test_loader, device, class_names)
    
    test_time = time.time() - test_start_time
    print(f"Tempo de teste: {test_time:.2f}s ({test_time/60:.2f} min)")
    
    print_metrics(metrics)
    
    # Print detailed classification report
    print("\n=== Relatorio de Classificacao Detalhado ===")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4, zero_division=0))
    
    # Save confusion matrix
    cm = np.array(metrics['confusion_matrix'])
    save_confusion_matrix(cm, class_names, os.path.join(results_dir, 'confusion_matrix.png'))
    
    # Save classification report
    report_path = os.path.join(results_dir, 'classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(classification_report(all_labels, all_preds, target_names=class_names, digits=4, zero_division=0))
    print(f"Relatorio de classificacao salvo em: {report_path}")
    
    # Save training curves if history available (WITHOUT the FE/FT divider line)
    history = metadata.get('history', {})
    if history:
        epochs_fe = metadata.get('epochs_fe', 30)
        save_training_curves(history, os.path.join(results_dir, 'training_curves.png'), epochs_fe)
    
    # Save prediction examples
    if args.save_examples and test_dataset is not None:
        save_prediction_examples(
            model, test_dataset, all_labels, all_preds, class_names,
            results_dir, args.num_examples
        )
    
    # Save ALL wrong predictions
    wrong_dir = os.path.join(results_dir, 'wrong_predictions')
    if test_dataset is not None:
        save_wrong_predictions(test_dataset, all_labels, all_preds, class_names, wrong_dir)
    
    # Save test time
    time_path = os.path.join(results_dir, 'test_time.txt')
    with open(time_path, 'w', encoding='utf-8') as f:
        f.write(f"Tempo de teste: {test_time:.2f}s ({test_time/60:.2f} min)\n")
        f.write(f"Total de imagens: {len(all_labels)}\n")
        f.write(f"Predicoes corretas: {sum(1 for p, l in zip(all_preds, all_labels) if p == l)}\n")
        f.write(f"Predicoes incorretas: {sum(1 for p, l in zip(all_preds, all_labels) if p != l)}\n")
    print(f"Tempo de teste salvo em: {time_path}")
    
    # Add error count to metrics
    wrong_count = sum(1 for p, l in zip(all_preds, all_labels) if p != l)
    total_count = len(all_labels)
    metrics['wrong_predictions'] = f"Erros: {wrong_count}/{total_count}"
    
    # Save all metrics
    metrics['metadata'] = metadata
    save_results(metrics, os.path.join(results_dir, 'metrics.json'))
    
    # Save run info so each experiment folder is self-describing
    run_info = {
        'model_path': os.path.abspath(args.model),
        'model_name': model_name,
        'run': run_folder,
        'results_dir': os.path.abspath(results_dir),
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'seed': args.seed,
        'batch_size': args.batch_size,
        'data_dir': os.path.abspath(args.data_dir),
    }
    save_results(run_info, os.path.join(results_dir, 'run_info.json'))
    
    print(f"\n=== Avaliacao Concluida ===")
    print(f"Tempo de teste: {test_time:.2f}s ({test_time/60:.2f} min)")
    print(f"Erros: {wrong_count}/{total_count}")
    print(f"Resultados salvos em: {results_dir}/")
    print(f"Log salvo em: {os.path.join(log_dir, 'test.log')}")
    
    # Restore stdout and close log
    sys.stdout = sys.__stdout__
    log_handle.close()


if __name__ == '__main__':
    main()