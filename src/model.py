import torch
import torch.nn as nn
from torchvision import models
from typing import List, Optional


def create_mobilenetv2_model(
    num_classes: int = 43,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    dropout_rate: float = 0.2,
    hidden_size: int = 512
) -> nn.Module:
    """
    Create MobileNetV2 model with custom classifier.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights
        freeze_backbone: Whether to freeze backbone parameters
        dropout_rate: Dropout rate for classifier
        hidden_size: Hidden layer size in classifier
    
    Returns:
        MobileNetV2 model with custom classifier
    """
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
    
    model.classifier = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(model.last_channel, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(hidden_size, num_classes)
    )
    
    return model


def unfreeze_blocks(model: nn.Module, block_indices: List[int]) -> None:
    """
    Unfreeze specific blocks of MobileNetV2 backbone.
    
    MobileNetV2 has 18 inverted residual blocks in features (indices 0-17).
    Block 0 is the first conv layer, blocks 1-17 are inverted residual blocks.
    
    Args:
        model: MobileNetV2 model
        block_indices: List of block indices to unfreeze (0-17)
    """
    for idx in block_indices:
        if idx < 0 or idx >= len(model.features):
            print(f"Aviso: Indice de bloco {idx} fora do intervalo (0-{len(model.features)-1})")
            continue
        for param in model.features[idx].parameters():
            param.requires_grad = True
        print(f"Bloco {idx} descongelado")


def freeze_all_backbone(model: nn.Module) -> None:
    """Freeze all backbone parameters."""
    for param in model.features.parameters():
        param.requires_grad = False


def unfreeze_all_backbone(model: nn.Module) -> None:
    """Unfreeze all backbone parameters (full fine-tuning)."""
    for param in model.features.parameters():
        param.requires_grad = True


def unfreeze_last_n_blocks(model: nn.Module, n: int) -> None:
    """Unfreeze the last n blocks of the backbone."""
    total_blocks = len(model.features)
    start_idx = max(0, total_blocks - n)
    block_indices = list(range(start_idx, total_blocks))
    unfreeze_blocks(model, block_indices)


def get_trainable_params(model: nn.Module, lr_backbone: float, lr_classifier: float):
    """
    Get parameter groups with different learning rates for fine-tuning.
    
    Args:
        model: MobileNetV2 model
        lr_backbone: Learning rate for unfrozen backbone parameters
        lr_classifier: Learning rate for classifier parameters
    
    Returns:
        List of parameter groups for optimizer
    """
    backbone_params = [p for p in model.features.parameters() if p.requires_grad]
    classifier_params = list(model.classifier.parameters())
    
    param_groups = []
    
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': lr_backbone})
    
    param_groups.append({'params': classifier_params, 'lr': lr_classifier})
    
    return param_groups


def print_trainable_layers(model: nn.Module) -> None:
    """Print which layers are trainable."""
    print("\n=== Camadas Treinaveis ===")
    for name, param in model.named_parameters():
        status = "TREINAVEL" if param.requires_grad else "CONGELADO"
        print(f"  {name}: {status}")