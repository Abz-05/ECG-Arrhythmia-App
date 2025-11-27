import torch
import torch.nn as nn

class CNN_LSTM_Arrhythmia(nn.Module):
    """
    Hybrid CNN-LSTM model for ECG arrhythmia classification.
    
    Architecture:
    1. CNN blocks extract spatial/morphological features (P-wave, QRS, T-wave shapes)
    2. LSTM captures temporal dependencies and rhythm patterns
    3. Fully connected layers classify into 5 AAMI arrhythmia classes
    
    Input: (batch_size, 1, 187) - single-lead ECG signals
    Output: (batch_size, 5) - class logits
    """
    
    def __init__(self, num_classes: int = 5, lstm_hidden: int = 64, lstm_layers: int = 1):
        super(CNN_LSTM_Arrhythmia, self).__init__()
        
        # ========================================
        # CNN FEATURE EXTRACTOR
        # ========================================
        # Learns local morphological patterns in ECG waveform
        
        self.cnn = nn.Sequential(
            # Block 1: Initial feature extraction
            nn.Conv1d(in_channels=1, out_channels=64, 
                     kernel_size=5, padding=2),  # Keep same length
            nn.BatchNorm1d(64),  # Stabilize training
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 187 -> 93
            nn.Dropout(0.2),
            
            # Block 2: Deeper feature extraction
            nn.Conv1d(in_channels=64, out_channels=128, 
                     kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 93 -> 46
            nn.Dropout(0.2),
            
            # Block 3: High-level feature extraction
            nn.Conv1d(in_channels=128, out_channels=128, 
                     kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 46 -> 23
            nn.Dropout(0.2)
        )
        
        # ========================================
        # LSTM TEMPORAL MODELER
        # ========================================
        # Captures rhythm and temporal dependencies
        # Bidirectional: learns patterns both forward and backward in time
        
        self.lstm = nn.LSTM(
            input_size=128,  # CNN output channels
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,  # Input shape: (batch, seq, feature)
            bidirectional=True,  # Learn both directions
            dropout=0.2 if lstm_layers > 1 else 0
        )
        
        # ========================================
        # CLASSIFICATION HEAD
        # ========================================
        # Maps LSTM features to class predictions
        
        # Bidirectional LSTM doubles the hidden size
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (Tensor): Input ECG signals, shape (batch, 1, 187)
            
        Returns:
            logits (Tensor): Class logits, shape (batch, 5)
        """
        # ========================================
        # STEP 1: CNN Feature Extraction
        # ========================================
        # Input: (batch, 1, 187)
        x = self.cnn(x)
        # Output: (batch, 128, 23) - 128 features at 23 time positions
        
        # ========================================
        # STEP 2: Reshape for LSTM
        # ========================================
        # LSTM expects (batch, sequence_length, features)
        # CNN outputs (batch, channels, length)
        # So we permute: (batch, channels, length) -> (batch, length, channels)
        x = x.permute(0, 2, 1)  # (batch, 23, 128)
        
        # ========================================
        # STEP 3: LSTM Temporal Modeling
        # ========================================
        # Process the sequence of CNN features
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (batch, 23, lstm_hidden*2) - outputs at each time step
        # h_n: (num_layers*2, batch, lstm_hidden) - final hidden states
        
        # Take the output from the last time step for classification
        x = lstm_out[:, -1, :]  # (batch, lstm_hidden*2)
        
        # ========================================
        # STEP 4: Classification
        # ========================================
        x = self.fc(x)  # (batch, num_classes)
        
        return x
    
    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ========================================
# MODEL INSTANTIATION HELPER
# ========================================

def create_model(num_classes: int = 5, device: str = 'cuda') -> CNN_LSTM_Arrhythmia:
    """
    Create and initialize the model.
    
    Args:
        num_classes (int): Number of output classes (default: 5)
        device (str): Device to place model on ('cuda' or 'cpu')
        
    Returns:
        model: Initialized CNN-LSTM model
    """
    model = CNN_LSTM_Arrhythmia(num_classes=num_classes)
    model = model.to(device)
    
    print(f"✓ Model created with {model.count_parameters():,} trainable parameters")
    print(f"✓ Model placed on device: {device}")
    
    return model