import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict

from dataset import ECGDataset
from model import create_model, CNN_LSTM_Arrhythmia
from utils import Config, PROJECT_ROOT

def get_balanced_class_weights(dataset: ECGDataset, balance_factor: float = 0.7) -> torch.Tensor:
    """
    Calculate BALANCED class weights to avoid over-predicting minority classes.
    
    Args:
        dataset: ECGDataset instance
        balance_factor: Float between 0-1. Lower = less aggressive balancing
                        0.5 = moderate, 0.7 = balanced (recommended), 1.0 = fully inverse
    
    Returns:
        weights (Tensor): Class weights for loss function
    """
    class_counts = dataset.get_class_distribution()
    total_samples = len(dataset)
    num_classes = len(class_counts)
    
    # Calculate raw weights (inverse frequency)
    raw_weights = []
    for class_id in sorted(class_counts.keys()):
        weight = total_samples / (num_classes * class_counts[class_id])
        raw_weights.append(weight)
    
    raw_weights = np.array(raw_weights)
    
    # Apply balance factor (reduces extreme weights)
    # Formula: balanced_weight = 1 + (raw_weight - 1) * balance_factor
    balanced_weights = 1 + (raw_weights - 1) * balance_factor
    
    weights = torch.tensor(balanced_weights, dtype=torch.float32)
    
    print("\n=== BALANCED CLASS WEIGHTS ===")
    for class_id, weight in enumerate(weights):
        count = class_counts[class_id]
        print(f"Class {class_id}: weight = {weight:.4f} (count = {count:,})")
    
    return weights


def get_sampler_weights(dataset: ECGDataset, balance_factor: float = 0.5) -> WeightedRandomSampler:
    """
    Create sample weights for WeightedRandomSampler with moderate balancing.
    """
    class_counts = dataset.get_class_distribution()
    
    # Calculate weights per sample
    sample_weights = []
    for label in dataset.labels:
        # Use softer weighting for sampler
        weight = 1.0 / (class_counts[label] ** balance_factor)
        sample_weights.append(weight)
    
    sample_weights = torch.tensor(sample_weights, dtype=torch.float64)
    sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    print("✓ Created BALANCED WeightedRandomSampler")
    
    return sampler


def train_epoch(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: str) -> Tuple[float, float]:
    """Train for one epoch with gradient clipping"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    
    for signals, labels in pbar:
        signals = signals.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(signals)
        loss = criterion(outputs, labels)
        
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item() * signals.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: str) -> Tuple[float, float]:
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for signals, labels in tqdm(val_loader, desc='Validation'):
            signals = signals.to(device)
            labels = labels.to(device)
            
            outputs = model(signals)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * signals.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def main():
    # Training Configuration
    BATCH_SIZE = 256
    LEARNING_RATE = 0.0005
    NUM_EPOCHS = 80
    PATIENCE = 15
    MODEL_SAVE_PATH = Config.MODEL_DIR / 'best_model_improved.pth'

    Config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Using device: {Config.DEVICE}")
    print(f"✓ Batch size: {BATCH_SIZE}")
    print(f"✓ Learning rate: {LEARNING_RATE}")
    
    # ========================================
    # LOAD DATASETS
    # ========================================
    print("\n[1/6] Loading datasets...")
    train_dataset = ECGDataset(Config.TRAIN_CSV, normalize=True)
    test_dataset = ECGDataset(Config.TEST_CSV, normalize=True)
    
    # ========================================
    # BALANCED CLASS WEIGHTS
    # ========================================
    print("\n[2/6] Calculating balanced class weights...")
    
    # Use BALANCED weights (less aggressive than before)
    class_weights = get_balanced_class_weights(train_dataset, balance_factor=0.6)
    class_weights = class_weights.to(Config.DEVICE)
    
    # Use MODERATE sampling
    train_sampler = get_sampler_weights(train_dataset, balance_factor=0.4)
    
    # ========================================
    # DATA LOADERS
    # ========================================
    print("\n[3/6] Creating data loaders...")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True if Config.DEVICE == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if Config.DEVICE == 'cuda' else False
    )
    
    print(f"✓ Training batches: {len(train_loader)}")
    print(f"✓ Test batches: {len(test_loader)}")
    
    # ========================================
    # CREATE MODEL
    # ========================================
    print("\n[4/6] Creating model...")
    model = create_model(num_classes=Config.NUM_CLASSES, device=Config.DEVICE)
    
    # ========================================
    # LOSS AND OPTIMIZER
    # ========================================
    print("\n[5/6] Setting up training components...")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Use AdamW (Adam with weight decay for better generalization)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Cosine annealing scheduler (smooth learning rate decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )
    
    # ========================================
    # TRAINING LOOP WITH EARLY STOPPING
    # ========================================
    print(f"\n[6/6] Training for up to {NUM_EPOCHS} epochs (with early stopping)...")
    
    best_val_acc = 0.0
    patience_counter = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, test_loader, criterion, Config.DEVICE
        )
        
        # Update learning rate
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, MODEL_SAVE_PATH)
            print(f"  ✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{PATIENCE}")
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\n⏹ Early stopping triggered after {epoch+1} epochs")
            print(f"Best validation accuracy: {best_val_acc:.2f}%")
            break
    
    # ========================================
    # PLOT TRAINING HISTORY
    # ========================================
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print("="*60)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss (Improved)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy (Improved)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Config.MODEL_DIR / 'training_history_improved.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✓ Training curves saved to {Config.MODEL_DIR / 'training_history_improved.png'}")


if __name__ == '__main__':
    main()