import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    accuracy_score, precision_recall_fscore_support
)
from pathlib import Path
from typing import Tuple, List

from dataset import ECGDataset
from model import create_model
from utils import Config, PROJECT_ROOT

# ========================================
# EVALUATION FUNCTION
# ========================================

def evaluate_model(model: nn.Module, test_loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model and return predictions and ground truth.
    
    Returns:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_probs: Prediction probabilities
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("Evaluating model on test set...")
    
    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(device)
            labels = labels.to(device)
            
            # Get predictions
            outputs = model(signals)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            # Store results
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ========================================
# PLOT CONFUSION MATRIX
# ========================================

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]):
    """
    Plot a detailed confusion matrix.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize by true labels (row-wise)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_xlabel('Predicted Label', fontsize=12)
    
    # Plot 2: Normalized (percentages)
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax2, cbar_kws={'label': 'Proportion'})
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=12)
    ax2.set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(Config.MODEL_DIR / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Confusion matrix saved to {Config.MODEL_DIR / 'confusion_matrix.png'}")


# ========================================
# COMPUTE DETAILED METRICS
# ========================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]):
    """
    Compute and display detailed classification metrics.
    """
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    
    # Print overall metrics
    print("\n" + "="*70)
    print("OVERALL PERFORMANCE")
    print("="*70)
    print(f"Overall Accuracy: {accuracy*100:.2f}%")
    print(f"Total Samples: {len(y_true):,}")
    
    # Print per-class metrics
    print("\n" + "="*70)
    print("PER-CLASS METRICS")
    print("="*70)
    print(f"{'Class':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-"*70)
    
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<25} {precision[i]:<12.4f} {recall[i]:<12.4f} "
              f"{f1[i]:<12.4f} {support[i]:<10,}")
    
    # Weighted averages
    precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    print("-"*70)
    print(f"{'Weighted Average':<25} {precision_avg:<12.4f} {recall_avg:<12.4f} "
          f"{f1_avg:<12.4f} {len(y_true):<10,}")
    print("="*70)
    
    # Scikit-learn classification report
    print("\nDETAILED CLASSIFICATION REPORT:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))


# ========================================
# ANALYZE ERRORS
# ========================================

def analyze_errors(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]):
    """
    Analyze common misclassification patterns.
    """
    print("\n" + "="*70)
    print("ERROR ANALYSIS")
    print("="*70)
    
    # Find all misclassifications
    errors = y_true != y_pred
    error_indices = np.where(errors)[0]
    
    print(f"\nTotal Errors: {len(error_indices):,} / {len(y_true):,} "
          f"({len(error_indices)/len(y_true)*100:.2f}%)")
    
    # Most common misclassification pairs
    from collections import Counter
    error_pairs = [(y_true[i], y_pred[i]) for i in error_indices]
    most_common = Counter(error_pairs).most_common(10)
    
    print("\nTop 10 Most Common Misclassifications:")
    print(f"{'True Class':<25} {'Predicted As':<25} {'Count':<10}")
    print("-"*70)
    
    for (true_class, pred_class), count in most_common:
        print(f"{class_names[true_class]:<25} {class_names[pred_class]:<25} {count:<10,}")
    
    print("="*70)


# ========================================
# MAIN EVALUATION
# ========================================

def main():
    print("="*70)
    print("ECG ARRHYTHMIA CLASSIFICATION - MODEL EVALUATION")
    print("="*70)
    
    # Check if model exists
    model_path = Config.MODEL_DIR / 'best_model_improved.pth'
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        print("Please train the model first using train.py")
        return
    
    print(f"✓ Using device: {Config.DEVICE}")
    
    # ========================================
    # STEP 1: LOAD TEST DATA
    # ========================================
    print("\n[1/5] Loading test dataset...")
    test_dataset = ECGDataset(Config.TEST_CSV, normalize=True)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=2
    )
    
    # ========================================
    # STEP 2: LOAD MODEL
    # ========================================
    print("\n[2/5] Loading trained model...")
    model = create_model(num_classes=Config.NUM_CLASSES, device=Config.DEVICE)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Model loaded from epoch {checkpoint['epoch']+1}")
    print(f"✓ Validation accuracy during training: {checkpoint['val_acc']:.2f}%")
    
    # ========================================
    # STEP 3: EVALUATE
    # ========================================
    print("\n[3/5] Running evaluation...")
    y_true, y_pred, y_probs = evaluate_model(model, test_loader, Config.DEVICE)
    
    # ========================================
    # STEP 4: COMPUTE METRICS
    # ========================================
    print("\n[4/5] Computing metrics...")
    compute_metrics(y_true, y_pred, Config.CLASS_NAMES)
    
    # ========================================
    # STEP 5: VISUALIZATIONS
    # ========================================
    print("\n[5/5] Creating visualizations...")
    
    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred, Config.CLASS_NAMES)
    
    # Error analysis
    analyze_errors(y_true, y_pred, Config.CLASS_NAMES)
    
    # ========================================
    # SAVE PREDICTIONS
    # ========================================
    print("\n" + "="*70)
    print("Saving predictions...")
    
    # Save predictions to file
    predictions_file = Config.MODEL_DIR / 'predictions.npz'
    np.savez(predictions_file, 
             y_true=y_true, 
             y_pred=y_pred, 
             y_probs=y_probs)
    
    print(f"✓ Predictions saved to {predictions_file}")
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()