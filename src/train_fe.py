"""
Feature Extraction Training Script

Trains only the classifier head of MobileNetV2 with frozen backbone.
This script should be run ONCE to create the base model for all fine-tuning experiments.
"""
import argparse
import copy
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

# Add src to path so sibling modules (dataset, model, utils) can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import get_data_loaders, DEFAULT_DATA_DIR, DEFAULT_BATCH_SIZE
from model import create_mobilenetv2_model
from utils import (
    set_seed, get_device, TimeTracker, save_model, print_model_info,
    count_parameters, train_one_epoch, validate,
)
from config import (
    NUM_CLASSES, EPOCHS_FE, LR_FE, DEFAULT_SEED,
)

# Hyperparameters
BATCH_SIZE = DEFAULT_BATCH_SIZE
DATA_DIR = DEFAULT_DATA_DIR
SEED = DEFAULT_SEED

MODEL_SAVE_PATH = "models/fe_model.pth"
TIME_SAVE_PATH = "models/fe_time.json"
METADATA_SAVE_PATH = "models/fe_metadata.json"


def main():
    parser = argparse.ArgumentParser(description='Feature Extraction Training')
    parser.add_argument('--data-dir', type=str, default=DATA_DIR, help='Data directory')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--epochs', type=int, default=EPOCHS_FE, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=LR_FE, help='Learning rate')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    parser.add_argument('--model-save', type=str, default=MODEL_SAVE_PATH, help='Model save path')
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    device = get_device()
    print(f"Dispositivo: {device}")
    
    # Track time
    time_tracker = TimeTracker()
    time_tracker.start('total_training')
    time_tracker.start('data_loading')
    
    # Data loaders
    train_loader, val_loader, class_names = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )
    
    time_tracker.stop('data_loading')
    time_tracker.start('model_creation')
    
    # Create model
    model = create_mobilenetv2_model(
        num_classes=NUM_CLASSES,
        pretrained=True,
        freeze_backbone=True
    )
    model = model.to(device)
    
    print_model_info(model, "Feature Extraction Model")
    print(f"Modelo criado com sucesso!")
    time_tracker.stop('model_creation')
    time_tracker.start('training')
    
    # Loss and optimizer (only classifier parameters)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Training loop
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"\n{'='*60}")
    print(f"FASE 1: Feature Extraction")
    print(f"Epocas: {args.epochs} | LR: {args.lr}")
    print(f"{'='*60}\n")
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoca {epoch}/{args.epochs}")
        print("-" * 40)
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Treino  - Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val     - Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"*** Novo melhor modelo salvo (acc: {best_acc:.4f}) ***")
    
    time_tracker.stop('training')
    
    # Load best weights
    model.load_state_dict(best_model_wts)
    
    # Save model
    metadata = {
        'num_classes': NUM_CLASSES,
        'epochs_fe': args.epochs,
        'lr_fe': args.lr,
        'batch_size': args.batch_size,
        'best_val_acc': best_acc,
        'history': history,
        'class_names': class_names,
        'seed': args.seed,
    }
    
    save_model(model, args.model_save, metadata)
    time_tracker.save(TIME_SAVE_PATH)
    
    # Save metadata separately for easy access
    import json
    with open(METADATA_SAVE_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    time_tracker.print_summary()
    print(f"\nFeature Extraction concluido!")
    print(f"Melhor acuracia de validacao: {best_acc:.4f}")
    print(f"Modelo salvo em: {args.model_save}")
    print(f"Tempos salvos em: {TIME_SAVE_PATH}")
    print(f"Metadados salvos em: {METADATA_SAVE_PATH}")


if __name__ == '__main__':
    main()
