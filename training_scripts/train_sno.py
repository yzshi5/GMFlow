"""
Training script for Super Resolution Operator model.
This script trains a neural network to perform super-resolution on 3D data.

The file should be run from the exp/ directory.
"""

# Standard library imports
import os
import sys
import glob
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from abc import ABC, abstractmethod

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn.functional as F

# Add parent directory to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# -----------------------------------------------------------------------------
# PATH CONFIG (edit once here; optional env vars override these defaults)
# -----------------------------------------------------------------------------
# Example:
#   export GMFLOW_DATA_ROOT="/path/to/your/data_root"
#   export GMFLOW_OUTPUT_ROOT="/path/to/your/output_root"
DATA_ROOT = Path(os.getenv('GMFLOW_DATA_ROOT', 'path/to/your/data_root'))
OUTPUT_ROOT = Path(os.getenv('GMFLOW_OUTPUT_ROOT', 'path/to/your/output_root'))
SNO_RUN_NAME = os.getenv('GMFLOW_SNO_RUN_NAME', 'path/to/your/sno_run_name')
NORM_DATA_SUBDIR = Path('norm_1c_final')
DEFAULT_SNO_LOW_RES_DIRS = [
    'norm_m6_128_64_48_fmax_075_clean',
    'norm_m7_128_64_48_fmax_075_clean',
    'norm_m44_128_64_48_fmax_075_clean',
]
DEFAULT_SNO_HIGH_RES_DIRS = [
    'norm_m6_256_128_clean',
    'norm_m7_256_128_clean',
    'norm_m44_256_128_clean',
]

# Local imports (these may show linting warnings but are resolved at runtime)
from utils.super_resolution_operator import *


# =============================================================================
# CONFIGURATION CLASSES
# =============================================================================

class abstract:
    """Abstract base class for model configurations."""
    
    class ModelConfig(ABC):
        """Abstract base class for model configurations."""
        
        @abstractmethod
        def to_dict(self) -> dict:
            """Convert configuration to dictionary."""
            pass


@dataclass
class ModelConfig(abstract.ModelConfig):
    """
    Configuration class for Super Resolution Operator model training.
    
    This dataclass contains all the parameters needed for model initialization,
    training, and data loading.
    """
    
    # Data dimensions
    n_x: int = 128
    n_y: int = 64
    n_t: int = 48
    n_chan: int = 4

    # Model architecture parameters
    width_en: int = 24
    in_width: int = 4  # Should match n_chan
    last_conv_model_time: int = 24 #32
    
    # Training parameters
    batch_size: int = 2
    epochs: int = 100
    learning_rate: float = 5e-4 # final around 1e-6
    save_interval: int = 50
    
    # Optimizer parameters
    optimizer_type: str = "AdamW"
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)  # Default betas for Adam/AdamW
    
    # Scheduler parameters
    scheduler_type: str = "CosineAnnealingLR"
    T_max: int = 100
    
    # Loss function parameters
    loss_type: str = "LpLoss"
    loss_p: int = 2
    size_average: bool = True
    
    # Device configuration
    device: str = "cuda:1"
    num_workers: int = 4
    pin_memory: bool = True
    
    # Paths
    encoder_path: str = str(OUTPUT_ROOT / SNO_RUN_NAME)
    data_paths_low: List[str] = field(default_factory=lambda: [
        str(DATA_ROOT / NORM_DATA_SUBDIR / data_dir_name)
        for data_dir_name in DEFAULT_SNO_LOW_RES_DIRS
    ])
    data_paths_high: List[str] = field(default_factory=lambda: [
        str(DATA_ROOT / NORM_DATA_SUBDIR / data_dir_name)
        for data_dir_name in DEFAULT_SNO_HIGH_RES_DIRS
    ])
    
    # Logging parameters
    enable_logging: bool = True
    
    # Model saving parameters
    save_model: bool = True

    # Data scaling parameters
    data_scale: float = 5.0

    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        self._setup_device()
        self._create_directories()
    
    def _setup_device(self):
        """Setup device configuration."""
        if self.device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            self.device = "cpu"
            self.pin_memory = False
    
    def _create_directories(self):
        """Create necessary directories."""
        if 'path/to/your' in str(DATA_ROOT) or 'path/to/your' in str(OUTPUT_ROOT):
            raise ValueError(
                'Please set DATA_ROOT/OUTPUT_ROOT in this file or via env vars '
                '(GMFLOW_DATA_ROOT, GMFLOW_OUTPUT_ROOT) before running.'
            )
        Path(self.encoder_path).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'data_dims': [self.n_x, self.n_y, self.n_t, self.n_chan],
            'model_params': {
                'width_en': self.width_en,
                'in_width': self.in_width,
                'last_conv_model_time': self.last_conv_model_time
            },
            'training_params': {
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'learning_rate': self.learning_rate,
                'save_interval': self.save_interval
            },
            'optimizer_params': {
                'type': self.optimizer_type,
                'weight_decay': self.weight_decay,
                'betas': self.betas
            },
            'scheduler_params': {
                'type': self.scheduler_type,
                'step_size': self.step_size,
                'gamma': self.gamma
            },
            'loss_params': {
                'type': self.loss_type,
                'p': self.loss_p,
                'size_average': self.size_average
            },
            'device': self.device,
            'paths': {
                'encoder_path': self.encoder_path,
                'data_paths_low': self.get_low_res_paths(),
                'data_paths_high': self.get_high_res_paths()
            }
        }
    
    def get_dims(self) -> List[int]:
        """Get data dimensions as a list."""
        return [self.n_x, self.n_y, self.n_t]
    
    def get_model_input_width(self) -> int:
        """Get model input width (including grid coordinates)."""
        return self.in_width + 3  # +3 for grid coordinates
    
    def get_optimizer(self, model_parameters):
        """Create optimizer based on configuration."""
        if self.optimizer_type == "Adam":
            return optim.Adam(
                model_parameters, 
                lr=self.learning_rate, 
                weight_decay=self.weight_decay,
                betas=self.betas
            )
        elif self.optimizer_type == "AdamW":
            return optim.AdamW(
                model_parameters, 
                lr=self.learning_rate, 
                weight_decay=self.weight_decay,
                betas=self.betas
            )
        elif self.optimizer_type == "RMSprop":
            return optim.RMSprop(
                model_parameters, 
                lr=self.learning_rate, 
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_type}")
    
    def get_scheduler(self, optimizer):
        """Create scheduler based on configuration."""
        if self.scheduler_type == "StepLR":
            return optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=self.step_size, 
                gamma=self.gamma
            )
        elif self.scheduler_type == "ExponentialLR":
            return optim.lr_scheduler.ExponentialLR(
                optimizer, 
                gamma=self.gamma
            )
        elif self.scheduler_type == "CosineAnnealingLR":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.epochs
            )
        elif self.scheduler_type == "None":
            return None
        else:
            raise ValueError(f"Unsupported scheduler: {self.scheduler_type}")
    
    def get_loss_function(self):
        """Create loss function based on configuration."""
        if self.loss_type == "LpLoss":
            return LpLoss(p=self.loss_p, size_average=self.size_average)
        elif self.loss_type == "MixedLoss":
            return MixedLoss(alpha=0.5, size_average=self.size_average)
        elif self.loss_type == "MSELoss":
            return nn.MSELoss()
        elif self.loss_type == "L1Loss":
            return nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_type}")
    
    def print_config(self):
        """Print configuration summary."""
        print("="*60)
        print("MODEL CONFIGURATION")
        print("="*60)
        print(f"Data dimensions: {self.get_dims()}")
        print(f"Model width: {self.width_en}")
        print(f"Input width: {self.get_model_input_width()}")
        print(f"Batch size: {self.batch_size}")
        print(f"Epochs: {self.epochs}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Data scale: {self.data_scale}")
        print(f"Device: {self.device}")
        print(f"Optimizer: {self.optimizer_type}")
        print(f"Scheduler: {self.scheduler_type}")
        print(f"Loss function: {self.loss_type}")
        print("Low-res data directories:")
        for path in self.get_low_res_paths():
            print(f"  - {path}")
        print("High-res data directories:")
        for path in self.get_high_res_paths():
            print(f"  - {path}")
        print("="*60)

    def get_low_res_paths(self) -> List[str]:
        """Normalize low-resolution data paths to a list."""
        return self._ensure_list(self.data_paths_low)

    def get_high_res_paths(self) -> List[str]:
        """Normalize high-resolution data paths to a list."""
        return self._ensure_list(self.data_paths_high)

    @staticmethod
    def _ensure_list(path_like) -> List[str]:
        """Ensure the returned value is a list of strings."""
        if isinstance(path_like, (list, tuple)):
            return list(path_like)
        return [path_like]


# =============================================================================
# INITIALIZE CONFIGURATION
# =============================================================================

# Create configuration instance
config = ModelConfig()

# Print configuration
config.print_config()

# =============================================================================
# MODEL INITIALIZATION
# =============================================================================

# Initialize model
model_encoder = SuperResolutionOperator(
    in_width=config.get_model_input_width(), 
    width=config.width_en,
    last_conv_model_time=config.last_conv_model_time
).to(config.device)

# Count parameters
nn_params = sum(p.numel() for p in model_encoder.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {nn_params:,}")

# Initialize optimizer, scheduler, and loss function using config
optimizer = config.get_optimizer(model_encoder.parameters())
scheduler = config.get_scheduler(optimizer)
myloss = config.get_loss_function()

print(f"Optimizer: {config.optimizer_type}")
print(f"Scheduler: {config.scheduler_type}")
print(f"Loss function: {config.loss_type}")



# =============================================================================
# DATASET CLASSES
# =============================================================================

class SimDataset(Dataset):
    """
    Dataset class for loading simulation data from .npy files.

    Args:
        root_dir (str): Directory containing the .npy files
        transform (callable, optional): Optional transform to be applied on a sample
        file_pattern (str): Pattern to match files (default: "*.npy")
        scale (float): Scaling factor to normalize the first three channels (default: 1.0), here I set it to 5.0 
    """
    
    def __init__(self, root_dir, transform=None, file_pattern="*.npy", scale=5.0):
        self.root_dir = root_dir
        self.transform = transform
        self.scale = scale

        # Get all .npy files, filtering out hidden and empty files
        all_files = [
            f for f in os.listdir(root_dir)
            if f.endswith('.npy') and not f.startswith('._') and not f.startswith('.')
        ]

        # Create full paths and filter out empty files
        self.file_list = []
        for f in all_files:
            full_path = os.path.join(root_dir, f)
            # Check if file is non-empty
            if os.path.getsize(full_path) > 0:
                self.file_list.append(full_path)

        if not self.file_list:
            raise FileNotFoundError(f"No valid .npy files found in {root_dir}")

        # Sort files by numerical value in filename
        self.file_list.sort(key=self._extract_sample_index)

        print(f"Found {len(self.file_list)} valid files in {root_dir}")
    
    @staticmethod
    def _extract_sample_index(path: str) -> int:
        """Extract numeric index from filename, defaulting to 0 if not present."""
        basename = os.path.splitext(os.path.basename(path))[0]
        match = re.search(r'(?:sim[_-]?)(\d+)$', basename)
        if match:
            return int(match.group(1))
        # fallback to trailing digits anywhere
        match = re.search(r'(\d+)$', basename)
        return int(match.group(1)) if match else 0

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load numpy array from file
        sample = np.load(self.file_list[idx])

        # Convert to tensor
        sample = torch.from_numpy(sample).float()

        # Normalize the first three channels
        if sample.shape[0] >= 3:
            sample[:3] = sample[:3] / self.scale

        if self.transform:
            sample = self.transform(sample)

        return sample


class CombinedDataset(Dataset):
    """
    Dataset class that combines two datasets (e.g., input and target).
    
    Args:
        dataset1 (Dataset): First dataset (e.g., low-resolution data)
        dataset2 (Dataset): Second dataset (e.g., high-resolution data)
    """
    
    def __init__(self, dataset1, dataset2):
        len1 = len(dataset1)
        len2 = len(dataset2)
        if len1 != len2:
            min_len = min(len1, len2)
            print(f"Warning: dataset length mismatch ({len1} vs {len2}). "
                  f"Using the first {min_len} samples from each dataset.")
        else:
            min_len = len1
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.length = min_len

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.dataset1[idx]
        y = self.dataset2[idx]
        return x, y


# =============================================================================
# DATA LOADING
# =============================================================================

print("Loading datasets...")
low_res_paths = config.get_low_res_paths()
high_res_paths = config.get_high_res_paths()

if len(low_res_paths) != len(high_res_paths):
    raise ValueError("The number of low-resolution paths must match high-resolution paths.")

paired_datasets = []
total_samples = 0
for idx, (low_dir, high_dir) in enumerate(zip(low_res_paths, high_res_paths), start=1):
    print(f"\nPair {idx}:")
    print(f"  Low-res path : {low_dir}")
    print(f"  High-res path: {high_dir}")
    low_dataset = SimDataset(root_dir=low_dir, scale=config.data_scale)
    high_dataset = SimDataset(root_dir=high_dir, scale=config.data_scale)
    pair_dataset = CombinedDataset(dataset1=low_dataset, dataset2=high_dataset)
    paired_datasets.append(pair_dataset)
    total_samples += len(pair_dataset)
    print(f"  Samples in pair {idx}: {len(pair_dataset)}")

if len(paired_datasets) == 1:
    combined_dataset = paired_datasets[0]
else:
    combined_dataset = ConcatDataset(paired_datasets)

print(f"\nTotal combined dataset size: {total_samples}")

# Create data loader
loader_tr = DataLoader(
    combined_dataset, 
    batch_size=config.batch_size, 
    shuffle=True, 
    num_workers=config.num_workers,
    pin_memory=config.pin_memory
)
print(f"Data loader created with batch size: {config.batch_size}")

# =============================================================================
# TRAINING
# =============================================================================

print("Starting training...")
print(f"Training for {config.epochs} epochs")
print(f"Model will be saved to: {config.encoder_path}")

# Create log file for training if logging is enabled
log_file = None
if config.enable_logging:
    from datetime import datetime
    log_file = Path(config.encoder_path) / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    print(f"Training log will be saved to: {log_file}")

# Train the model with logging
train_encoder(
    model=model_encoder,
    optimizer=optimizer,
    train_loader=loader_tr,
    epochs=config.epochs,
    my_loss=myloss,
    scheduler=scheduler,
    saved_model=config.save_model,
    save_int=config.save_interval,
    save_path=Path(config.encoder_path),
    device=config.device,
    log_file=str(log_file) if log_file else None  # Enable logging if configured
)

print("Training completed!")




# =============================================================================
# MODEL EVALUATION
# =============================================================================

def evaluate_model(model_path, data_loader, device):
    """
    Evaluate the trained model on a sample batch.
    
    Args:
        model_path (Path): Path to the saved model
        data_loader (DataLoader): Data loader for evaluation
        device (str): Device to run evaluation on
    """
    print("\nEvaluating trained model...")
    
    # Load the trained model
    model_eval = SuperResolutionOperator(
        in_width=config.get_model_input_width(), 
        width=config.width_en,
        last_conv_model_time=config.last_conv_model_time
    ).to(device)
    model_eval.load_state_dict(torch.load(model_path))
    model_eval.eval()
    
    # Get a test batch
    test_batch, target_batch = next(iter(data_loader))
    test_batch = test_batch.to(device)
    target_batch = target_batch.to(device)
    
    # Run inference
    with torch.no_grad():
        output_batch = model_eval(test_batch)
    
    # Print results
    print(f"Input shape: {test_batch.shape}")
    print(f"Output shape: {output_batch.shape}")
    print(f"Target shape: {target_batch.shape}")
    
    # Calculate reconstruction error
    reconstruction_error = torch.mean(torch.abs(output_batch - target_batch))
    print(f"Mean absolute reconstruction error: {reconstruction_error:.6f}")
    
    return output_batch, target_batch, reconstruction_error


if __name__ == "__main__":
    # Evaluate the final model
    final_model_path = Path(config.encoder_path) / f'Encoder_epoch_{config.epochs}.pt'
    
    if final_model_path.exists():
        evaluate_model(final_model_path, loader_tr, config.device)
    else:
        print(f"Model file not found: {final_model_path}")
        print("Available model files:")
        for model_file in Path(config.encoder_path).glob('Encoder_epoch_*.pt'):
            print(f"  - {model_file.name}")
            
        # Evaluate the latest available model
        model_files = list(Path(config.encoder_path).glob('Encoder_epoch_*.pt'))
        if model_files:
            latest_model = max(model_files, key=lambda x: int(x.stem.split('_')[-1]))
            print(f"\nEvaluating latest model: {latest_model.name}")
            evaluate_model(latest_model, loader_tr, config.device)

