#!/usr/bin/env python
# coding: utf-8

"""
Training and evaluation script for Flow Matching model on latent space data.
This script trains a UNet conditional model with white noise prior and evaluates it using MMD and SWD metrics.
"""


# white noise + standard diffusion unet 
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from utils.latent_ofm_clean_pred import OFMModel
from utils.unet_ofm import UNet_cond
from utils.metrics import unbiased_mmd2_torch, swd_stable

# =============================================================================
# CONFIGURATION
# =============================================================================

# Data dimensions
n_x = 32
n_y = 16
n_t = 16
n_chan = 1

dims = [n_x, n_y, n_t]
dims_all = [n_chan, n_x, n_y, n_t]

# Device and paths
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# -----------------------------------------------------------------------------
# PATH CONFIG (edit once here; optional env vars override these defaults)
# -----------------------------------------------------------------------------
# Example:
#   export GMFLOW_DATA_ROOT="/path/to/your/data_root"
#   export GMFLOW_OUTPUT_ROOT="/path/to/your/output_root"
DATA_ROOT = Path(os.getenv('GMFLOW_DATA_ROOT', 'path/to/your/data_root'))
OUTPUT_ROOT = Path(os.getenv('GMFLOW_OUTPUT_ROOT', 'path/to/your/output_root'))
RUN_NAME = 'latent_ofm_rupture_128M'

# Input files
LATENT_TRAIN_PATH = DATA_ROOT / 'latent_data' / 'mid_075_128_rupture_128M.npy'
SRC_LOCS_PATH = DATA_ROOT / 'rupture_hypo_all.npy'

# Output directory
spath = OUTPUT_ROOT / RUN_NAME
spath.mkdir(parents=True, exist_ok=True)
saved_model = True

# Model hyperparameters (scaled-up ~32M parameter UNet)
model_config = {
    'hidden_channels': 96,
    'num_res_blocks': 2,
    'num_heads': 8,
    'attention_res': '8,4',
    'channel_mult': (1, 2, 3, 4)
}

# Training parameters
alpha = -1
epochs = 300
sigma_min = 1e-4
N = 1
batch_size = 64
sample_n_eval = 20
sample_method = 'euler'

# =============================================================================
# DATA LOADING
# =============================================================================

# Load training data
#x_train = np.load('/oak-data/yshi5/GNO_GANO/training_data/BA_simulation/latent_data/mid_low_10000_04_128_test2.npy')
if 'path/to/your' in str(DATA_ROOT) or 'path/to/your' in str(OUTPUT_ROOT):
    raise ValueError(
        'Please set DATA_ROOT/OUTPUT_ROOT in this file or via env vars '
        '(GMFLOW_DATA_ROOT, GMFLOW_OUTPUT_ROOT) before running.'
    )

x_train = np.load(LATENT_TRAIN_PATH)
x_train = torch.Tensor(x_train)

scale = 1.0 # std = 0.7, keep 1.0 for samplicity
x_train = x_train * scale 

# Load source locations (conditions)
src_locs_np = np.load(SRC_LOCS_PATH)

# Expect last channel to encode region id (m6, m7, m44) appended during concatenation
if src_locs_np.shape[1] < 4:
    raise ValueError(f"Expected rupture_hypo_all.npy to have at least 4 channels, got shape {src_locs_np.shape}")

spatial = torch.from_numpy(src_locs_np[:, :3]).float() / 10000.0
region_channel = torch.from_numpy(src_locs_np[:, 3:4]).float()

# Optional normalization for region channel: scale to roughly [-1, 1] range
region_channel_normalized = region_channel / 10.0

# Concatenate to form conditioning tensor with 4 channels
src_locs = torch.cat([spatial, region_channel_normalized], dim=1)

# Verify data consistency
assert len(x_train) == len(src_locs), f"Data length mismatch: x_train has {len(x_train)} samples, src_locs has {len(src_locs)} samples"

print(f"Loaded {len(x_train)} training samples")
print(f"Data shape: {x_train.shape}")
print(f"Condition shape: {src_locs.shape}")

# Create dataset and data loader
train_dataset = TensorDataset(x_train, src_locs)
loader_tr = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# =============================================================================
# TRAINING
# =============================================================================

print("="*60)
print("Starting Training")
print("="*60)

for i in range(N):
    model = UNet_cond(
        dims=dims_all, 
        conds_channels=4, 
        **model_config
    ).to(device)
    
    # Calculate and print model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Information:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=2e-4,
        weight_decay=1e-4,
        betas=(0.9, 0.95)
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6
    )
    
    fmot = OFMModel(model, alpha=alpha, sigma_min=sigma_min, device=device, dims=dims)
    fmot.train(
        loader_tr, 
        optimizer, 
        epochs=epochs, 
        scheduler=scheduler, 
        eval_int=0, 
        save_int=300, 
        generate=False, 
        save_path=spath,
        saved_model=saved_model
    )

print("\nTraining completed!")

# =============================================================================
# EVALUATION: Use MMD and SWD metrics
# =============================================================================

print("\n" + "="*60)
print("Starting Evaluation")
print("="*60)

# Load the trained model checkpoint
checkpoint_path = spath / f'epoch_{epochs}.pt'
if not checkpoint_path.exists():
    # Try to find the latest checkpoint
    checkpoint_files = list(spath.glob('epoch_*.pt'))
    if checkpoint_files:
        checkpoint_path = max(checkpoint_files, key=lambda x: int(x.stem.split('_')[-1]))
        print(f"Found checkpoint: {checkpoint_path}")
    else:
        raise FileNotFoundError(f"No checkpoint found in {spath}")

print(f"Loading model from: {checkpoint_path}")

# Initialize model architecture (same as training)
model_eval = UNet_cond(
    dims=dims_all, 
    conds_channels=4, 
    **model_config
).to(device)

# Load checkpoint (saved as model.state_dict() directly)
model_eval.load_state_dict(torch.load(checkpoint_path, map_location=device))
model_eval.eval()

# Create OFMModel wrapper for sampling
fmot_eval = OFMModel(model_eval, alpha=alpha, sigma_min=sigma_min, device=device, dims=dims)

# Get number of training samples
n_train_samples = len(x_train)
print(f"Number of training samples: {n_train_samples}")

# Generate samples with the same conditions as training data
print("\nGenerating samples from trained model...")
print(f"Using batch size for generation: {batch_size}")

generated_samples = []
train_samples_for_eval = []

# Generate samples in batches
with torch.no_grad():
    for batch_idx in range(0, n_train_samples, batch_size):
        end_idx = min(batch_idx + batch_size, n_train_samples)
        batch_conds = src_locs[batch_idx:end_idx].to(device)
        batch_size_actual = end_idx - batch_idx
        
        # Generate samples with conditions
        batch_generated = fmot_eval.sample(
            dims=dims,
            conds=batch_conds,
            n_channels=n_chan,
            n_samples=batch_size_actual,
            n_eval=sample_n_eval,
            method=sample_method,
        )
        
        # Get corresponding training samples
        batch_train = x_train[batch_idx:end_idx].to(device)
        
        generated_samples.append(batch_generated.cpu())
        train_samples_for_eval.append(batch_train.cpu())
        
        if (batch_idx // batch_size + 1) % 10 == 0:
            print(f"Generated {end_idx}/{n_train_samples} samples...")

# Concatenate all batches
generated_samples = torch.cat(generated_samples, dim=0)
train_samples_for_eval = torch.cat(train_samples_for_eval, dim=0)

print(f"\nGenerated samples shape: {generated_samples.shape}")
print(f"Training samples shape: {train_samples_for_eval.shape}")

# =============================================================================
# Compute Metrics: MMD and SWD
# =============================================================================

print("\n" + "="*60)
print("Computing Metrics")
print("="*60)

# Compute MMD
print("\nComputing MMD (Maximum Mean Discrepancy)...")
mmd_value = unbiased_mmd2_torch(
    X=generated_samples,
    Y=train_samples_for_eval,
    gamma=None,  # Use default gamma=1.0
    device=device
)

print(f"MMD: {mmd_value:.6f}")

# Compute SWD (Sliced Wasserstein Distance)
print("\nComputing SWD (Sliced Wasserstein Distance)...")
swd_value = swd_stable(
    X=generated_samples,
    Y=train_samples_for_eval,
    n_runs=10,
    n_proj=256
)

print(f"SWD: {swd_value:.6f}")

# =============================================================================
# Save Results
# =============================================================================

results = {
    'mmd': float(mmd_value),
    'swd': float(swd_value),
    'n_samples': n_train_samples,
    'checkpoint_path': str(checkpoint_path),
    'model_config': {
        'dims': dims,
        'network': model_config,
        'alpha': alpha,
        'sigma_min': sigma_min
    }
}

results_path = spath / f'evaluation_results_epoch_{epochs}.txt'
with open(results_path, 'w') as f:
    f.write("="*60 + "\n")
    f.write("Evaluation Results\n")
    f.write("="*60 + "\n\n")
    f.write(f"Checkpoint: {checkpoint_path}\n")
    f.write(f"Number of samples: {n_train_samples}\n\n")
    f.write("Metrics:\n")
    f.write(f"  MMD: {mmd_value:.6f}\n")
    f.write(f"  SWD: {swd_value:.6f}\n\n")
    f.write("Model Configuration:\n")
    for key, value in results['model_config'].items():
        f.write(f"  {key}: {value}\n")

print(f"\nResults saved to: {results_path}")

# Save generated samples for later use
generated_samples_path = spath / f'generated_samples_epoch_{epochs}.npy'
np.save(generated_samples_path, generated_samples.numpy())
print(f"Generated samples saved to: {generated_samples_path}")

print("\n" + "="*60)
print("Evaluation Complete!")
print("="*60)
print(f"\nSummary:")
print(f"  MMD: {mmd_value:.6f}")
print(f"  SWD: {swd_value:.6f}")
print("="*60)
