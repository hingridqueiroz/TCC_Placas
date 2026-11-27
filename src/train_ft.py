"""
Fine-Tuning Independente - TCC

Cada experimento começa DO MESMO modelo FE (models/fe_model.pth).
NÃO há transferência de pesos entre experimentos.
NÃO há acumulação, NÃO há progressão.

Dois modos MUTUAMENTE EXCLUSIVOS (escolha apenas UM por execução):
1. --full           : Full Fine-Tuning clássico (descongela TODO o backbone)
2. --last-n N       : Fine-Tuning parcial - últimos N blocos
3. --first-n N      : Fine-Tuning parcial - primeiros N blocos
4. --blocks ...     : Fine-Tuning parcial - blocos específicos

NÃO combina múltiplos experimentos em uma execução.
"""
import argparse
import copy
import json
import os
import sys
from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim

# Add src to path so sibling modules (dataset, model, utils) can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import get_data_loaders, DEFAULT_DATA_DIR, DEFAULT_BATCH_SIZE
from model import (
    create_mobilenetv2_model, unfreeze_blocks, unfreeze_all_backbone,
    get_trainable_params
)
from utils import (
    set_seed, get_device, TimeTracker, save_model, load_model, count_parameters,
    train_one_epoch, validate,
)
from config import (
    NUM_CLASSES, EPOCHS_FT, LR_FT_BACKBONE, LR_FT_CLASSIFIER, DEFAULT_SEED,
    PATIENCE_FT,
)

# Hyperparameters
BATCH_SIZE = DEFAULT_BATCH_SIZE
DATA_DIR = DEFAULT_DATA_DIR
SEED = DEFAULT_SEED
PATIENCE = PATIENCE_FT

FE_MODEL_PATH = "models/fe_model.pth"
RESULTS_BASE_DIR = "results_train/ft_independent"
LOGS_BASE_DIR = "logs/independent"


def get_unfrozen_blocks(model) -> List[int]:
    return [
        i for i, module in enumerate(model.features)
        if any(param.requires_grad for param in module.parameters())
    ]


def create_optimizer(model, lr_backbone, lr_classifier):
    param_groups = get_trainable_params(model, lr_backbone, lr_classifier)
    return optim.Adam(param_groups)


def train_experiment(
    model,
    train_loader,
    val_loader,
    criterion,
    device,
    lr_backbone,
    lr_classifier,
    epochs,
    patience,
    exp_name,
    time_tracker,
    mode: str,
    blocks: List[int],
):
    blocos = get_unfrozen_blocks(model)
    trainable_count = count_parameters(model)['trainable']
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENTO INDEPENDENTE: {exp_name}")
    print(f"MODO: {mode}")
    print(f"Blocos treináveis: {blocos}")
    print(f"Total de blocos: {len(blocos)}")
    print(f"Parâmetros treináveis: {trainable_count:,}")
    print(f"{'='*60}\n")
    
    optimizer = create_optimizer(model, lr_backbone, lr_classifier)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    counter = 0
    
    time_tracker.start(exp_name)
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoca {epoch}/{epochs}")
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
            counter = 0
            print(f"*** Novo melhor modelo salvo (acc: {best_acc:.4f}) ***")
        else:
            counter += 1
            print(f"Sem melhora: {counter}/{patience}")
        
        if counter >= patience:
            print(f"\nEarly Stopping na epoca {epoch}")
            break
    
    exp_time = time_tracker.stop(exp_name)
    
    model.load_state_dict(best_model_wts)
    
    best_epoch = history['val_acc'].index(best_acc) + 1 if history['val_acc'] else 0
    epochs_trained = len(history['train_loss'])
    
    print(f"\nExperimento {exp_name} concluido!")
    print(f"  Melhor acuracia: {best_acc:.4f} (epoca {best_epoch})")
    print(f"  Epocas treinadas: {epochs_trained}")
    print(f"  Tempo: {exp_time:.2f}s ({exp_time/60:.2f} min)")
    
    return {
        'best_acc': best_acc,
        'best_model_wts': best_model_wts,
        'history': history,
        'epochs_trained': epochs_trained,
        'best_epoch': best_epoch,
        'exp_time': exp_time,
        'trainable_params': count_parameters(model)['trainable'],
        'current_blocks': get_unfrozen_blocks(model),
    }


def get_experiment_folder_name(mode: str, blocks: List[int]) -> str:
    """Gera nome da pasta do experimento."""
    if mode == "FULL":
        return "independent_full"
    else:
        if len(blocks) == 1:
            return f"independent_block_{blocks[0]}"
        elif blocks == list(range(min(blocks), max(blocks) + 1)):
            # Sequência contígua
            return f"independent_blocks_{min(blocks)}_to_{max(blocks)}"
        else:
            # Blocos específicos não contíguos
            blocks_str = "_".join(str(b) for b in sorted(blocks))
            return f"independent_blocks_{blocks_str}"


def run_independent_ft(args):
    # Determinar qual experimento rodar (APENAS UM)
    if args.full:
        exp_name = "full"
        blocks = list(range(19))  # todos os 19 módulos features (0-18)
        mode = "FULL"
        is_full = True
    elif args.last_n is not None:
        n = min(args.last_n, 17)
        blocks = list(range(18 - n, 18))
        exp_name = f"last_{n}"
        mode = "PARTIAL"
        is_full = False
    elif args.first_n is not None:
        n = min(args.first_n, 17)
        blocks = list(range(1, n + 1))
        exp_name = f"first_{n}"
        mode = "PARTIAL"
        is_full = False
    elif args.blocks is not None:
        blocks = args.blocks
        exp_name = "custom"
        mode = "PARTIAL"
        is_full = False
    else:
        print("Erro: Escolha exatamente UM: --full, --last-n, --first-n ou --blocks")
        sys.exit(1)
    
    # Validar blocos (apenas para PARTIAL)
    if mode == "PARTIAL":
        for b in blocks:
            if b < 1 or b > 17:
                print(f"Erro: Bloco {b} inválido. Deve ser entre 1 e 17.")
                sys.exit(1)
    
    # Nome da pasta do experimento
    exp_folder = get_experiment_folder_name("FULL" if args.full else "PARTIAL", blocks)
    
    # Diretórios do experimento
    exp_results_dir = os.path.join(RESULTS_BASE_DIR, exp_folder)
    exp_logs_dir = os.path.join(LOGS_BASE_DIR, exp_folder)
    os.makedirs(exp_results_dir, exist_ok=True)
    os.makedirs(exp_logs_dir, exist_ok=True)
    
    # Log file
    log_file = os.path.join(exp_logs_dir, f"{exp_name}.log")
    
    # Redirecionar stdout para arquivo de log também
    import sys

    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, data):
            for f in self.files:
                f.write(data)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    
    log_file_handle = open(log_file, 'w', encoding='utf-8')
    sys.stdout = Tee(sys.stdout, log_file_handle)
    
    try:
        set_seed(args.seed)
        device = get_device()
        print(f"Dispositivo: {device}")
        print(f"Experimento: {exp_name} | MODO: {'FULL' if args.full else 'PARTIAL'} | blocos: {blocks}")
        print(f"Pasta resultados: {exp_results_dir}")
        print(f"Pasta logs: {exp_logs_dir}")
        
        # Carregar modelo FE UMA VEZ e guardar pesos
        print(f"\nCarregando modelo FE de: {args.fe_model}")
        fe_model = create_mobilenetv2_model(
            num_classes=NUM_CLASSES,
            pretrained=False,
            freeze_backbone=True
        )
        fe_model = fe_model.to(device)
        metadata_fe = load_model(fe_model, args.fe_model, device)
        fe_best_model_wts = copy.deepcopy(fe_model.state_dict())
        print(f"Modelo FE carregado (val_acc: {metadata_fe.get('best_val_acc', 'N/A'):.4f})")
        
        train_loader, val_loader, class_names = get_data_loaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size
        )
        
        criterion = nn.CrossEntropyLoss()
        os.makedirs(exp_results_dir, exist_ok=True)
        
        total_tracker = TimeTracker()
        total_tracker.start('total_experiments')
        
        # Modelo NOVO para ESTE experimento
        model = create_mobilenetv2_model(
            num_classes=NUM_CLASSES,
            pretrained=False,
            freeze_backbone=True
        )
        model = model.to(device)
        
        # SEMPRE carrega pesos do FE original
        model.load_state_dict(fe_best_model_wts)
        
        # Classifier sempre treinável
        for param in model.classifier.parameters():
            param.requires_grad = True
        
        # Descongelar conforme o modo
        is_full = args.full
        if is_full:
            # FULL: descongela TODO o backbone (model.features)
            unfreeze_all_backbone(model)
        else:
            # PARTIAL: descongela apenas blocos selecionados
            unfreeze_blocks(model, blocks)
        
        # Treinar
        exp_result = train_experiment(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=nn.CrossEntropyLoss(),
            device=device,
            lr_backbone=args.lr_backbone,
            lr_classifier=args.lr_classifier,
            epochs=args.epochs,
            patience=PATIENCE,
            exp_name=exp_name,
            time_tracker=total_tracker,
            mode="FULL" if args.full else "PARTIAL",
            blocks=blocks,
        )
        
        # Salvar na pasta do experimento
        model_path = os.path.join(exp_results_dir, f"model.pth")
        time_path = os.path.join(exp_results_dir, "time.json")
        meta_path = os.path.join(exp_results_dir, "metadata.json")
        history_path = os.path.join(exp_results_dir, "history.json")
        
        exp_metadata = {
            'experiment_name': exp_name,
            'mode': "FULL" if args.full else "PARTIAL",
            'is_full': args.full,
            'blocks_unfrozen': blocks,
            'num_unfrozen_blocks': len(blocks),
            'trainable_params': exp_result['trainable_params'],
            'best_val_acc': exp_result['best_acc'],
            'best_epoch': exp_result['best_epoch'],
            'epochs_trained': exp_result['epochs_trained'],
            'history': exp_result['history'],
            'exp_time': exp_result['exp_time'],
            'lr_backbone': args.lr_backbone,
            'lr_classifier': args.lr_classifier,
            'batch_size': args.batch_size,
            'seed': args.seed,
            'fe_model_source': args.fe_model,
            'fe_model_acc': metadata_fe.get('best_val_acc'),
            'data_dir': os.path.abspath(args.data_dir),
            'train_images': len(train_loader.dataset),
            'val_images': len(val_loader.dataset),
        }
        
        save_model(model, model_path, exp_metadata)
        
        with open(time_path, 'w') as f:
            json.dump({'exp_time': exp_result['exp_time']}, f, indent=2)
        
        with open(meta_path, 'w') as f:
            json.dump(exp_metadata, f, indent=2, default=str)
        
        with open(history_path, 'w') as f:
            json.dump(exp_result['history'], f, indent=2, default=str)
        
        # Copiar log para pasta de resultados também
        log_dest = os.path.join(exp_results_dir, "training.log")
        import shutil
        shutil.copy2(log_file, log_dest)
        
        print(f"\n  Modelo salvo: {model_path}")
        print(f"  Metadados: {meta_path}")
        print(f"  Histórico: {history_path}")
        print(f"  Log: {log_file}")
        
        total_time = total_tracker.stop('total_experiments')
        
        print(f"\n{'='*60}")
        print(f"FINE-TUNING INDEPENDENTE CONCLUIDO")
        print(f"Experimento: {exp_name} | MODO: {'FULL' if args.full else 'PARTIAL'}")
        print(f"Blocos: {blocks}")
        print(f"Parâmetros treináveis: {exp_result['trainable_params']:,}")
        print(f"Melhor val_acc: {exp_result['best_acc']:.4f}")
        print(f"Tempo: {exp_result['exp_time']:.2f}s ({exp_result['exp_time']/60:.2f} min)")
        print(f"Pasta resultados: {exp_results_dir}")
        print(f"Pasta logs: {exp_logs_dir}")
        print(f"{'='*60}")
        
        total_tracker.print_summary()
        
    finally:
        sys.stdout = sys.__stdout__
        log_file_handle.close()


def main():
    parser = argparse.ArgumentParser(description='Fine-Tuning Independente - TCC (um experimento por execução)')
    
    # APENAS UM por execução (mutuamente exclusivos)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--full', action='store_true',
                        help='FULL Fine-Tuning: descongela TODO o backbone (model.features)')
    group.add_argument('--last-n', type=int, default=None,
                        help='Últimos N blocos PARCIAIS (ex: 6 -> 12,13,14,15,16,17)')
    group.add_argument('--first-n', type=int, default=None,
                        help='Primeiros N blocos PARCIAIS (ex: 6 -> 1,2,3,4,5,6)')
    group.add_argument('--blocks', type=int, nargs='+', default=None,
                        help='Blocos específicos PARCIAIS (ex: 17 15 8)')
    
    # Configuração
    parser.add_argument('--data-dir', type=str, default=DATA_DIR, help='Data directory')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--epochs', type=int, default=EPOCHS_FT, help='Epochs por experimento')
    parser.add_argument('--lr-backbone', type=float, default=LR_FT_BACKBONE, help='Backbone LR')
    parser.add_argument('--lr-classifier', type=float, default=LR_FT_CLASSIFIER, help='Classifier LR')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    parser.add_argument('--fe-model', type=str, default=FE_MODEL_PATH, help='FE model path')
    args = parser.parse_args()
    
    if not os.path.exists(args.fe_model):
        print(f"Erro: Modelo FE não encontrado em {args.fe_model}")
        print("Execute primeiro: python src/train_fe.py")
        sys.exit(1)
    
    run_independent_ft(args)


if __name__ == '__main__':
    main()