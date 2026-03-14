"""
Training script for AutoEncoder model.
This script trains an autoencoder neural network for 3D data compression and reconstruction.

The file should be run from the exp/ directory.
"""

# Standard library imports
import os
import sys
import glob
import re
import argparse
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
import statsmodels.api as sm
from scipy.stats import binned_statistic

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
AENO_RUN_NAME = os.getenv('GMFLOW_AENO_RUN_NAME', 'path/to/your/aeno_run_name')
NORM_DATA_SUBDIR = Path('norm_1c_final')
DEFAULT_AENO_DATA_DIRS = [
    'norm_m6_128_64_48_fmax_075_clean',
    'norm_m7_128_64_48_fmax_075_clean',
    'norm_m44_128_64_48_fmax_075_clean',
]

# Local imports (these may show linting warnings but are resolved at runtime)
from utils.autoencoding_operator import *


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
class AutoEncoderConfig(abstract.ModelConfig):
    """
    Configuration class for AutoEncoder model training.
    
    This dataclass contains all the parameters needed for model initialization,
    training, and data loading.
    """
    
    # Data dimensions
    n_x: int = 128
    n_y: int = 64
    n_t: int = 48
    n_chan: int = 4
    
    # Model architecture parameters
    width_en: int = 32
    in_width: int = 4
    
    # Training parameters
    batch_size: int = 8
    epochs: int = 200
    learning_rate: float = 5e-4
    save_interval: int = 200
    
    # Optimizer parameters
    optimizer_type: str = "AdamW"
    weight_decay: float = 1e-4
    
    # Scheduler parameters
    scheduler_type: str = "CosineAnnealingLR"
    T_max: int = 200
    warmup_epochs: int = 0  # Number of epochs for warmup phase
    
    # Loss function parameters
    loss_type: str = "LpLoss"
    loss_p: int = 2
    size_average: bool = True
    
    # Device configuration
    device: str = "cuda:3"
    num_workers: int = 4
    pin_memory: bool = True
    
    # Paths 
    encoder_path: str = str(OUTPUT_ROOT / AENO_RUN_NAME)
    data_paths: List[str] = field(default_factory=lambda: [
        str(DATA_ROOT / NORM_DATA_SUBDIR / data_dir_name)
        for data_dir_name in DEFAULT_AENO_DATA_DIRS
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
            self.device = "cuda:7" if torch.cuda.is_available() else "cpu"
        
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
                'in_width': self.in_width
            },
            'training_params': {
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'learning_rate': self.learning_rate,
                'save_interval': self.save_interval
            },
            'optimizer_params': {
                'type': self.optimizer_type,
                'weight_decay': self.weight_decay
            },
            'scheduler_params': {
                'type': self.scheduler_type,
                'T_max': self.T_max,
                'warmup_epochs': self.warmup_epochs
            },
            'loss_params': {
                'type': self.loss_type,
                'p': self.loss_p,
                'size_average': self.size_average
            },
            'device': self.device,
            'paths': {
                'encoder_path': self.encoder_path,
                'data_paths': self.data_paths
            }
        }
    
    def get_dims(self) -> List[int]:
        """Get data dimensions as a list."""
        return [self.n_x, self.n_y, self.n_t]
    
    def get_model_input_width(self) -> int:
        """Get model input width (including grid coordinates)."""
        return self.in_width + 3  # +3 for grid coordinates

    def get_data_paths(self) -> List[str]:
        """Normalize data paths to a list."""
        return self._ensure_list(self.data_paths)

    @staticmethod
    def _ensure_list(path_like) -> List[str]:
        """Ensure the returned value is a list of strings."""
        if isinstance(path_like, (list, tuple)):
            return list(path_like)
        return [path_like]
    
    def get_optimizer(self, model_parameters):
        """Create optimizer based on configuration."""
        if self.optimizer_type == "Adam":
            return optim.Adam(
                model_parameters, 
                lr=self.learning_rate, 
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == "AdamW":
            return optim.AdamW(
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
        elif self.scheduler_type == "None":
            return None
        elif self.scheduler_type == "CosineAnnealingLR":
            if self.warmup_epochs > 0:
                # Create sequential scheduler with warmup + cosine annealing
                # Warmup: linear increase from 0 to initial_lr over warmup_epochs
                warmup_scheduler = optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.01,  # Start at 1% of initial learning rate
                    end_factor=1.0,    # End at 100% of initial learning rate
                    total_iters=self.warmup_epochs
                )
                
                # Cosine annealing: T_max should be (epochs - warmup_epochs)
                # to account for the warmup phase
                cosine_T_max = self.T_max - self.warmup_epochs
                if cosine_T_max <= 0:
                    raise ValueError(f"T_max ({self.T_max}) must be greater than warmup_epochs ({self.warmup_epochs})")
                
                cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=cosine_T_max
                )
                
                # Chain warmup and cosine annealing together
                return optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[self.warmup_epochs]
                )
            else:
                # No warmup, just cosine annealing
                return optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, 
                    T_max=self.T_max
                )
        elif self.scheduler_type == "LinearLR":
            # Linear scheduler that decreases LR linearly to 0
            if self.warmup_epochs > 0:
                # Create sequential scheduler with warmup + linear decay
                # Warmup: linear increase from 0 to initial_lr over warmup_epochs
                warmup_scheduler = optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.01,  # Start at 1% of initial learning rate
                    end_factor=1.0,    # End at 100% of initial learning rate
                    total_iters=self.warmup_epochs
                )
                
                # Linear decay: decrease from initial_lr to 0 over (epochs - warmup_epochs)
                decay_epochs = self.epochs - self.warmup_epochs
                if decay_epochs <= 0:
                    raise ValueError(f"epochs ({self.epochs}) must be greater than warmup_epochs ({self.warmup_epochs})")
                
                linear_decay_scheduler = optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=1.0,   # Start at 100% of initial learning rate
                    end_factor=0.0,    # End at 0% (LR decays to 0)
                    total_iters=decay_epochs
                )
                
                # Chain warmup and linear decay together
                return optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, linear_decay_scheduler],
                    milestones=[self.warmup_epochs]
                )
            else:
                # No warmup, just linear decay from initial_lr to 0
                return optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=1.0,   # Start at 100% of initial learning rate
                    end_factor=0.0,    # End at 0% (LR decays to 0)
                    total_iters=self.epochs
                )
        else:
            raise ValueError(f"Unsupported scheduler: {self.scheduler_type}")
    
    def get_loss_function(self):
        """Create loss function based on configuration."""
        if self.loss_type == "LpLoss":
            return LpLoss(p=self.loss_p, size_average=self.size_average)
        elif self.loss_type == "MSELoss":
            return nn.MSELoss()
        elif self.loss_type == "L1Loss":
            return nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_type}")
    
    def print_config(self):
        """Print configuration summary."""
        print("="*60)
        print("AUTOENCODER CONFIGURATION")
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
        if self.scheduler_type == "CosineAnnealingLR" and self.warmup_epochs > 0:
            print(f"  Warmup epochs: {self.warmup_epochs}")
            print(f"  Cosine annealing T_max: {self.T_max - self.warmup_epochs}")
        elif self.scheduler_type == "LinearLR":
            if self.warmup_epochs > 0:
                print(f"  Warmup epochs: {self.warmup_epochs}")
                print(f"  Linear decay epochs: {self.epochs - self.warmup_epochs}")
            else:
                print(f"  Linear decay epochs: {self.epochs} (LR decreases to 0)")
        print(f"Loss function: {self.loss_type}")
        print("Data directories:")
        for path in self.get_data_paths():
            print(f"  - {path}")
        print("="*60)


# =============================================================================
# INITIALIZE CONFIGURATION
# =============================================================================

# Create configuration instance
config = AutoEncoderConfig()

# Print configuration
config.print_config()


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
        scale (float): Scaling factor to normalize the first three channels (default: 5.0)
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


# =============================================================================
# MODEL INITIALIZATION
# =============================================================================

# Initialize model
model_encoder = AutoEncoderOperator(in_width=config.get_model_input_width(), width=config.width_en).to(config.device)

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
# DATA LOADING
# =============================================================================

print("Loading datasets...")
data_paths = config.get_data_paths()

datasets = []
total_samples = 0
for idx, data_path in enumerate(data_paths, start=1):
    print(f"Loading dataset {idx}: {data_path}")
    dataset = SimDataset(root_dir=data_path, scale=config.data_scale)
    datasets.append(dataset)
    total_samples += len(dataset)
    print(f"  Dataset {idx} size: {len(dataset)}")

if len(datasets) == 1:
    combined_dataset = datasets[0]
else:
    combined_dataset = ConcatDataset(datasets)

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

def run_training():
    """Run the autoencoder training loop once."""
    print("Starting training...")
    print(f"Training for {config.epochs} epochs")
    print(f"Model will be saved to: {config.encoder_path}")

    log_file = None
    if config.enable_logging:
        from datetime import datetime
        log_file = Path(config.encoder_path) / f"autoencoder_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        print(f"Training log will be saved to: {log_file}")

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
        log_file=str(log_file) if log_file else None,
    )

    print("Training completed!")


# =============================================================================
# MODEL EVALUATION
# =============================================================================

def evaluate_autoencoder(model_path, data_loader, device):
    """
    Evaluate the trained autoencoder model on a sample batch.
    
    Args:
        model_path (Path): Path to the saved model
        data_loader (DataLoader): Data loader for evaluation
        device (str): Device to run evaluation on
    """
    print("\nEvaluating trained autoencoder...")
    
    # Load the trained model
    model_eval = AutoEncoderOperator(in_width=config.get_model_input_width(), width=config.width_en).to(device)
    model_eval.load_state_dict(torch.load(model_path))
    model_eval.eval()
    
    # Get a test batch
    test_batch = next(iter(data_loader))
    test_batch = test_batch.to(device)
    
    # Run inference
    with torch.no_grad():
        # Test full reconstruction
        reconstructed = model_eval(test_batch)
        
        # Test encoding
        encoded = model_eval(test_batch, output_encode=True)
        
        # Test decoding from encoded representation
        decoded = model_eval(encoded, decode=True)
    
    # Print results
    print(f"Input shape: {test_batch.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Encoded shape: {encoded.shape}")
    print(f"Decoded shape: {decoded.shape}")
    
    # Calculate reconstruction error
    reconstruction_error = torch.mean(torch.abs(reconstructed - test_batch))
    print(f"Mean absolute reconstruction error: {reconstruction_error:.6f}")
    
    # Calculate compression ratio
    input_size = test_batch.numel()
    encoded_size = encoded.numel()
    compression_ratio = input_size / encoded_size
    print(f"Compression ratio: {compression_ratio:.2f}:1")
    
    return reconstructed, encoded, decoded, reconstruction_error


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train or evaluate AutoEncoder model')

    # Mode selection
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate'],
                       default='train', help='Mode: train or evaluate (default: train)')

    # Evaluation specific arguments
    parser.add_argument('--model_path', type=str, help='Path to model file for evaluation (optional, will use latest if not specified)')

    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    if args.mode == 'evaluate':
        # Evaluation mode - skip training
        print("Evaluation mode selected")

        # Determine model path
        if args.model_path:
            model_path = Path(args.model_path)
            if not model_path.exists():
                print(f"Specified model file not found: {model_path}")
                sys.exit(1)
        else:
            # Find latest model in default location
            model_files = list(Path(config.encoder_path).glob('AutoEncoder_epoch_*.pt'))
            if not model_files:
                print(f"No model files found in {config.encoder_path}")
                sys.exit(1)
            model_path = max(model_files, key=lambda x: int(x.stem.split('_')[-1]))
            print(f"Using latest model: {model_path.name}")

        # Evaluate the model
        evaluate_autoencoder(model_path, loader_tr, config.device)

    else:
        # Training mode (default)
        print("Training mode selected")

        run_training()

        # Evaluate the final model
        final_model_path = Path(config.encoder_path) / f'AutoEncoder_epoch_{config.epochs}.pt'

        if final_model_path.exists():
            evaluate_autoencoder(final_model_path, loader_tr, config.device)
        else:
            print(f"Model file not found: {final_model_path}")
            print("Available model files:")
            for model_file in Path(config.encoder_path).glob('AutoEncoder_epoch_*.pt'):
                print(f"  - {model_file.name}")

            # Evaluate the latest available model
            model_files = list(Path(config.encoder_path).glob('AutoEncoder_epoch_*.pt'))
            if model_files:
                latest_model = max(model_files, key=lambda x: int(x.stem.split('_')[-1]))
                print(f"\nEvaluating latest model: {latest_model.name}")
                evaluate_autoencoder(latest_model, loader_tr, config.device)


