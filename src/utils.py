import os
import sys
from pathlib import Path
import torch

def get_project_root() -> Path:
    """
    Determine the project root directory.
    Searches up from the current script's location until it finds 'src', 'models', or 'data'.
    """
    # Start from the directory of the calling script or current working directory
    try:
        current = Path(__file__).resolve().parent
    except NameError:
        current = Path(os.getcwd())

    for _ in range(5):  # Search up to 5 levels
        if (current / 'src').exists() or (current / 'models').exists() or (current / 'data').exists():
            return current
        current = current.parent
    
    # Fallback to current directory if not found
    return Path(os.getcwd())

# Global Project Root
PROJECT_ROOT = get_project_root()

# Ensure src is in python path
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

class Config:
    """Shared configuration constants."""
    
    # Paths
    DATA_DIR = PROJECT_ROOT / 'data'
    MODEL_DIR = PROJECT_ROOT / 'models'
    TRAIN_CSV = DATA_DIR / 'mitbih_train.csv'
    TEST_CSV = DATA_DIR / 'mitbih_test.csv'
    
    # Model Paths (Try multiple for robustness)
    POSSIBLE_MODEL_PATHS = [
        MODEL_DIR / 'best_model_improved.pth',
        MODEL_DIR / 'best_model.pth',
        Path('models/best_model_improved.pth'),
        Path('models/best_model.pth'),
    ]
    
    # Hardware
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Data Parameters
    NUM_CLASSES = 5
    SIGNAL_LENGTH = 187
    
    # Class Definitions
    CLASS_NAMES = [
        'Normal (N)',
        'Supraventricular (S)', 
        'Ventricular (V)',
        'Fusion (F)',
        'Unknown (Q)'
    ]
    
    CLASS_DESCRIPTIONS = {
        'Normal (N)': '✅ Normal sinus rhythm or bundle branch blocks. No immediate concern.',
        'Supraventricular (S)': '⚠️ Early beats originating above the ventricles. Usually benign but monitor.',
        'Ventricular (V)': '🚨 Premature ventricular contractions (PVCs). May require medical evaluation.',
        'Fusion (F)': 'ℹ️ Hybrid beat combining normal and ventricular activation. Rare pattern.',
        'Unknown (Q)': '❓ Paced beats or unclassifiable patterns. May need expert review.'
    }
    
    CLASS_COLORS = {
        0: '#2ecc71',  # Green for Normal
        1: '#f39c12',  # Orange for Supraventricular
        2: '#e74c3c',  # Red for Ventricular
        3: '#9b59b6',  # Purple for Fusion
        4: '#95a5a6'   # Gray for Unknown
    }
    
    CLASS_RISKS = {
        0: 'Low',
        1: 'Low-Medium',
        2: 'Medium-High',
        3: 'Medium',
        4: 'Variable'
    }

def find_model_path() -> Path:
    """Find the first existing model file from possible paths."""
    for path in Config.POSSIBLE_MODEL_PATHS:
        if path.exists():
            return path
    return None
