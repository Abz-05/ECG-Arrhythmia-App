class ECGDataset(Dataset):
    """
    Custom PyTorch Dataset for ECG signals.
    
    Each sample consists of:
    - Signal: 187-point ECG waveform (1D time series)
    - Label: Integer class (0-4) representing arrhythmia type
    """
    
    def __init__(self, csv_file: str, normalize: bool = True):
        """
        Args:
            csv_file (str): Path to CSV file with ECG data
            normalize (bool): Whether to apply z-score normalization per beat
        """
        # Load data: last column is label, rest are signal values
        self.data = pd.read_csv(csv_file, header=None)
        self.normalize = normalize
        
        # Separate features and labels
        self.signals = self.data.iloc[:, :-1].values.astype(np.float32)
        self.labels = self.data.iloc[:, -1].values.astype(np.int64)
        
        print(f"✓ Loaded {len(self.signals)} ECG samples from {csv_file}")
        
    def __len__(self) -> int:
        """Return total number of samples"""
        return len(self.signals)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single ECG sample and its label.
        
        Args:
            idx (int): Index of sample to retrieve
            
        Returns:
            signal (Tensor): Shape (1, 187) - single-channel ECG signal
            label (int): Class label (0-4)
        """
        # Extract signal and label
        signal = self.signals[idx].copy()
        label = self.labels[idx]
        
        # Normalize signal (z-score normalization per beat)
        # This ensures each beat has zero mean and unit variance
        if self.normalize:
            mean = signal.mean()
            std = signal.std()
            # Avoid division by zero
            if std > 1e-6:
                signal = (signal - mean) / std
        
        # Convert to PyTorch tensor and add channel dimension
        # Shape: (187,) -> (1, 187) for Conv1d compatibility
        signal = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
        
        return signal, label
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Return dictionary with class counts"""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique, counts))
