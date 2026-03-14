"""
End-to-end evaluation utilities for the Super-resolution Operator (SNO),
Autoencoding Operator (AENO), and Latent Operator Flow Matching (OFM) models.

This module consolidates the configuration details defined in the respective
training scripts and loads the datasets and pretrained checkpoints required for
joint evaluation.
"""
#%%
import sys
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple, Dict, Any

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import h5py

from utils.autoencoding_operator import AutoEncoderOperator
from utils.latent_ofm_clean_pred import OFMModel
from utils.super_resolution_operator import SuperResolutionOperator
from utils.unet_ofm import UNet_cond


# Ensure repo-local imports work when executing from `exp/`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# -----------------------------------------------------------------------------
# PATH CONFIG (edit once here; optional env vars override these defaults)
# -----------------------------------------------------------------------------
# Example:
#   export GMFLOW_DATA_ROOT="/path/to/your/data_root"
#   export GMFLOW_MODEL_ROOT="/path/to/your/model_root"
#   export GMFLOW_OUTPUT_ROOT="/path/to/your/output_root"
DATA_ROOT = Path(os.getenv("GMFLOW_DATA_ROOT", "path/to/your/data_root"))
MODEL_ROOT = Path(os.getenv("GMFLOW_MODEL_ROOT", "path/to/your/model_root"))
OUTPUT_ROOT = Path(os.getenv("GMFLOW_OUTPUT_ROOT", "path/to/your/output_root"))
TEMP_PLOT_ROOT = OUTPUT_ROOT / "temp_plot"
SAVE_PLOT_ROOT = OUTPUT_ROOT / "save_plot"


HYPOCENTER_PATH = DATA_ROOT / "rupture_hypo_test.npy"
BIAS_CURVE_PATH = MODEL_ROOT / "freq_bias" / "freq_mean_residual_075hz.npy"

MODEL_DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
ANALYSIS_DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
device = MODEL_DEVICE
#post_fix = "x_pred_unif_filter_05_t48_double" # 1c_add, 1c_add_tr
post_fix = "residual_calc_unbias" # 1c_add, 1c_add_tr
# Frequency-domain filtering defaults
freq_correction_ranges = [0.1, 0.96]

@dataclass
class SNOConfig:
    """Configuration replicated from `Train_Sup_rupture_04_10_128_256.py` (1C variant)."""

    n_x: int = 128
    n_y: int = 64
    n_t: int = 48
    n_chan: int = 4
    width_en: int = 24
    in_width: int = 4
    last_conv_model_time: int = 24
    device: str = device
    checkpoint_path: Path = MODEL_ROOT / "SNO" / "Encoder_epoch_100.pt"
    high_res_paths: Tuple[Path, ...] = (
        DATA_ROOT / "norm_1c_final" / "norm_m6_test_256_128_clean",
        DATA_ROOT / "norm_1c_final" / "norm_m7_test_256_128_clean",
        DATA_ROOT / "norm_1c_final" / "norm_m44_test_256_128_clean",
    )

    def model_input_width(self) -> int:
        return self.in_width + 3

    def spatial_shape(self) -> Tuple[int, int, int]:
        return self.n_x, self.n_y, self.n_t


@dataclass
class AENOConfig:
    """Configuration replicated from `Train_autoencoder_rupture_04_128_64_48.py` (1C variant)."""

    n_x: int = 128
    n_y: int = 64
    n_t: int = 48
    n_chan: int = 4
    width_en: int = 32
    in_width: int = 4
    device: str = device
    checkpoint_path: Path = MODEL_ROOT / "AENO" / "Encoder_epoch_200.pt"
    data_paths: Tuple[Path, ...] = (
        DATA_ROOT / "norm_1c_final" / "norm_m6_128_64_48_fmax_06_clean",
        DATA_ROOT / "norm_1c_final" / "norm_m7_128_64_48_fmax_06_clean",
        DATA_ROOT / "norm_1c_final" / "norm_m44_128_64_48_fmax_06_clean",
    )

    def model_input_width(self) -> int:
        return self.in_width + 3

    def spatial_shape(self) -> Tuple[int, int, int]:
        return self.n_x, self.n_y, self.n_t


@dataclass
class OFMConfig:
    """Configuration replicated from `latent_ofm_rupture_04_128.py`."""

    n_x: int = 32
    n_y: int = 16
    n_t: int = 16
    n_chan: int = 1
    width: int = 64
    sigma_min: float = 1e-4
    conds_channels: int = 4
    scale: float = 1.0 # scale the latent data by this factor
    device: str = device
    t_eps: float = 0.02 # try 0.02
    checkpoint_path: Path = MODEL_ROOT / "OFM" / "epoch_300.pt"
    latent_data_file: Path = DATA_ROOT / "latent_data" /  "mid_075_128_rupture_filter_res_t48.npy"
    model_config = {
    'hidden_channels': 96,
    'num_res_blocks': 2,
    'num_heads': 8,
    'attention_res': '8,4',
    'channel_mult': (1, 2, 3, 4)
    }
    def dims(self) -> List[int]:
        return [self.n_x, self.n_y, self.n_t]

    def dims_all(self) -> List[int]:
        return [self.n_chan, self.n_x, self.n_y, self.n_t]

SNO_CONFIG = SNOConfig()
AENO_CONFIG = AENOConfig()
OFM_CONFIG = OFMConfig()


class NumpyFileDataset(Dataset):
    """
    Lazily loads `.npy` files from one or multiple directories, mirroring the
    dataset utilities used in the training scripts.
    """

    def __init__(self, root_dirs: Sequence[Path] | Path, pattern: str = "*.npy", dtype=np.float32, scale=5.0):
        if isinstance(root_dirs, (list, tuple, set)):
            dirs = [Path(p) for p in root_dirs]
        else:
            dirs = [Path(root_dirs)]

        self.files: List[Path] = []
        for directory in dirs:
            if not directory.exists():
                raise FileNotFoundError(f"Directory not found: {directory}")

            dir_files = [
                filepath for filepath in directory.glob(pattern)
                if not (filepath.name.startswith(".") or filepath.name.startswith("._"))
            ]

            if not dir_files:
                print(f"WARNING: No files found in {directory} matching {pattern}")

            # Sort files within this directory by numeric suffix
            dir_files.sort(key=_numeric_suffix)
            self.files.extend(dir_files)

        if not self.files:
            joined = ", ".join(str(d) for d in dirs)
            raise FileNotFoundError(f"No files matching '{pattern}' found in {joined}")

        self.dtype = dtype
        self.scale = scale

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        data = np.load(self.files[idx])

        if data.shape[0] >= 3:
            data[:3] = data[:3] / self.scale
            
        return torch.from_numpy(data.astype(self.dtype, copy=False))

    def filenames(self) -> Sequence[Path]:
        return tuple(self.files)


def _numeric_suffix(filepath: Path) -> int:
    stem = filepath.stem
    parts = stem.split('_')
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    return 999999



def denormalize_displacement(data: torch.Tensor, sno_factor: float = 5.0) -> torch.Tensor:
    """
    Convert normalized rupture data back to original scale.

    Args:
        data: Tensor of shape [batch, 4, x, y, t] where channels 0-2 hold
              normalized displacement fields and channel 3 stores log10 scale.

    Returns:
        Tensor with channels 0-2 rescaled to physical magnitude.
    """
    if data.dim() != 5 or data.shape[1] < 4:
        raise ValueError(f"Expected data shape [batch, 4, ...], got {data.shape}")

    rescaled = data.clone()
    log_scale = rescaled[:, 3:4]  # [batch, 1, x, y, t]
    mean_log = log_scale.mean(dim=(2, 3, 4), keepdim=True)  # [batch, 1, 1, 1, 1]
    scale = torch.pow(10.0, mean_log)

    rescaled[:, :3] = rescaled[:, :3] * scale * sno_factor
    # only return the first 3 channels
    return rescaled[:, :3]



def apply_frequency_bias_correction(
    final_output: torch.Tensor,
    bias_curve_path: Path,
    sampling_freq: float = 4.0,
    mw_row: int = 3,
    fft_batch_size: int = 8,
) -> torch.Tensor:
    """Apply multiplicative frequency-domain bias correction to synthetic output."""
    if final_output.dim() != 5 or final_output.shape[1] != 3:
        raise ValueError(f"Expected final_output shape [batch, 3, H, W, T], got {final_output.shape}")

    bias_curves = np.load(bias_curve_path)
    if bias_curves.ndim != 2 or bias_curves.shape[0] <= mw_row:
        raise ValueError(f"Unexpected bias table shape {bias_curves.shape} in {bias_curve_path}")

    freq_ref = bias_curves[0].astype(np.float64, copy=False)
    bias_ref = bias_curves[mw_row].astype(np.float64, copy=False)

    n_t = final_output.shape[-1]
    freq_rfft = np.fft.rfftfreq(n_t, d=1.0 / sampling_freq)
    if freq_rfft.shape != freq_ref.shape:
        raise ValueError(
            f"Frequency bin count mismatch: rfft={freq_rfft.shape}, bias={freq_ref.shape}. "
            "Ensure T and sampling_freq match the bias file."
        )

    max_abs_diff = np.max(np.abs(freq_rfft - freq_ref))
    if max_abs_diff > 1e-6:
        raise ValueError(
            f"Frequency grids differ too much (max abs diff={max_abs_diff:.3e}). "
            "Ensure rfftfreq and bias file use the same T and sampling frequency."
        )
    if max_abs_diff > 0.0:
        bias_ref = np.interp(freq_rfft, freq_ref, bias_ref)
        freq_ref = freq_rfft

    fmin, fmax = freq_correction_ranges
    freq_mask = (freq_ref >= fmin) & (freq_ref <= fmax)
    bias_band_limited = np.where(freq_mask, bias_ref, 0.0)

    correction_factor = torch.exp(
        torch.from_numpy(bias_band_limited).to(device=final_output.device, dtype=final_output.dtype)
    ).view(1, 1, 1, 1, -1)

    if fft_batch_size <= 0:
        raise ValueError("fft_batch_size must be positive")

    corrected_chunks = []
    for start in range(0, final_output.shape[0], fft_batch_size):
        batch = final_output[start:start + fft_batch_size]
        spec = torch.fft.rfft(batch, dim=-1)
        spec.mul_(correction_factor)
        corrected_batch = torch.fft.irfft(spec, n=n_t, dim=-1)
        corrected_chunks.append(corrected_batch)

    return torch.cat(corrected_chunks, dim=0)


def select_bias_row_from_conditions(conditions: torch.Tensor) -> int:
    """Select bias row [mw44, mw6, mw7] from normalized condition vector."""
    if conditions.dim() != 2 or conditions.shape[1] < 4:
        raise ValueError(f"Expected conditions shape [batch, >=4], got {conditions.shape}")

    magnitude = float(conditions[0, 3].item() * 10.0)
    row_by_target = {4.4: 1, 6.0: 2, 7.0: 3}
    closest_target = min(row_by_target.keys(), key=lambda target: abs(magnitude - target))
    return row_by_target[closest_target]


def _downsample_axis(data: np.ndarray, target_size: int, axis: int) -> np.ndarray:
    """Downsample by integer stride along a single axis."""
    current = data.shape[axis]
    if target_size <= 0:
        raise ValueError("target_size must be positive")
    if current == target_size:
        return data.copy()
    if current % target_size != 0:
        raise ValueError(f"Axis {axis} size {current} is not divisible by {target_size}")
    stride = current // target_size
    slices = [slice(None)] * data.ndim
    slices[axis] = slice(None, None, stride)
    return data[tuple(slices)]


def downsample_spatial(data: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    if data.shape[1] < target_hw[0] or data.shape[2] < target_hw[1]:
        raise ValueError(
            f"Input spatial dims {data.shape[1:3]} smaller than target {target_hw}"
        )
    ds_x = _downsample_axis(data, target_hw[0], axis=1)
    ds_xy = _downsample_axis(ds_x, target_hw[1], axis=2)
    return ds_xy.astype(np.float32, copy=False)


def cal_gmean_gstd(wfs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate geometric mean and geometric std along dim=0.
    """
    log_wfs = torch.log(wfs)
    mean = torch.mean(log_wfs, dim=0)
    std = torch.std(log_wfs, dim=0)
    return torch.exp(mean).cpu(), torch.exp(std).cpu()


def cal_fourier_amplitude(wfs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """
    Calculate fourier amplitude spectrum (FAS).

    Returns:
        mean/std of horizontal FAS, mean/std of vertical FAS (geometric).
    """
    bias = 1e-10
    wfs_f = torch.abs(torch.fft.rfft(wfs, dim=-1)) + bias
    power_mean_f = torch.sqrt((wfs_f[:, 0, :] ** 2 + wfs_f[:, 1, :] ** 2) / 2)
    h_mean_f, h_std_f = cal_gmean_gstd(power_mean_f)
    v_mean_f, v_std_f = cal_gmean_gstd(torch.abs(wfs_f[:, 2, :]))
    return h_mean_f, h_std_f, v_mean_f, v_std_f


def load_latent_dataset(path: Path) -> torch.Tensor:
    """Load the latent dataset used to train OFM."""
    latent_array = np.load(path)
    return torch.from_numpy(latent_array).float()


def load_hypocenters(path: Path) -> torch.Tensor:
    """Load and normalize hypocenter conditioning vectors.scaled by constant for hypo and magnitude, (M6, M7, M44)"""
    src_locs_np= np.load(path)
    spatial = torch.from_numpy(src_locs_np[:, :3]).float() / 10000.0
    region_channel = torch.from_numpy(src_locs_np[:, 3:4]).float()

    # Optional normalization for region channel: scale to roughly [-1, 1] range
    region_channel_normalized = region_channel / 10.0 # magnitude

    # Concatenate to form conditioning tensor with 4 channels
    src_locs = torch.cat([spatial, region_channel_normalized], dim=1)

    return src_locs


def _ensure_checkpoint(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def _ensure_paths_configured() -> None:
    roots = [
        ("DATA_ROOT", DATA_ROOT),
        ("MODEL_ROOT", MODEL_ROOT),
        ("OUTPUT_ROOT", OUTPUT_ROOT),
    ]
    bad = [name for name, value in roots if "path/to/your" in str(value)]
    if bad:
        raise ValueError(
            "Please set required path roots in this file or via env vars: "
            "GMFLOW_DATA_ROOT, GMFLOW_MODEL_ROOT, GMFLOW_OUTPUT_ROOT. "
            f"Unset placeholders: {', '.join(bad)}"
        )


def load_sno_model(config: SNOConfig = SNO_CONFIG) -> SuperResolutionOperator:
    """Instantiate and load the pretrained Super-resolution Operator."""
    checkpoint = _ensure_checkpoint(config.checkpoint_path)
    model = SuperResolutionOperator(
        in_width=config.model_input_width(),
        width=config.width_en,
        last_conv_model_time=config.last_conv_model_time,
    ).to(config.device)
    state_dict = torch.load(checkpoint, map_location=config.device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_aeno_model(config: AENOConfig = AENO_CONFIG) -> AutoEncoderOperator:
    """Instantiate and load the pretrained Autoencoding Operator."""
    checkpoint = _ensure_checkpoint(config.checkpoint_path)
    model = AutoEncoderOperator(
        in_width=config.model_input_width(),
        width=config.width_en,
    ).to(config.device)
    state_dict = torch.load(checkpoint, map_location=config.device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_ofm_model(config: OFMConfig = OFM_CONFIG) -> OFMModel:
    """Instantiate and load the pretrained Latent Operator Flow Matching model."""
    checkpoint = _ensure_checkpoint(config.checkpoint_path)
    network = UNet_cond(
        dims=config.dims_all(),
        conds_channels=config.conds_channels,
        **config.model_config
    ).to(config.device)

    state_dict = torch.load(checkpoint, map_location=config.device, weights_only=True)
    network.load_state_dict(state_dict)
    network.eval()

    fmot = OFMModel(
        network,
        sigma_min=config.sigma_min,
        device=config.device,
        dims=config.dims(),
        t_eps=config.t_eps,
    )

    return fmot

# Instantiate dataset handles for downstream evaluation.
_ensure_paths_configured()
high_res_dataset = NumpyFileDataset(SNO_CONFIG.high_res_paths)
latent_dataset = load_latent_dataset(OFM_CONFIG.latent_data_file)
hypocenters = load_hypocenters(HYPOCENTER_PATH)

#%%
# Load pretrained models.
sno_model = load_sno_model()
aeno_model = load_aeno_model()
ofm_pipeline = load_ofm_model()

#%%

def end_to_end_test(
    sample_idx: int,
    n_samples: int = 100,
    batch_size: int = 8,
    skip_plots: bool = False,
    save_outputs: bool = True,
    return_data: bool = False,
) -> Dict[str, Any] | Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """
    Run end-to-end test: OFM -> AENO -> SNO pipeline

    Args:
        sample_idx (int): Index of the sample to use for conditions

    Returns:
        Dictionary with results and metrics
    """
    print(f"\nRunning end-to-end test on sample {sample_idx}")

    # Get hypocenter conditions for this sample
    if sample_idx >= len(hypocenters):
        raise ValueError(f"Sample index {sample_idx} is out of range. Max index: {len(hypocenters)-1}")

    conditions = hypocenters[sample_idx].unsqueeze(0).to(OFM_CONFIG.device)  # [1, n_conds]
    conditions_batch = conditions.repeat(n_samples, 1)
    
    # test
    print(f"conditions: {conditions}\n")
    print(f"Using conditions shape: {conditions.shape}")

    # Step 1: Generate latent sample using OFM
    print("Step 1: Generating latent sample with OFM...")
    with torch.no_grad():
        latent_dims = OFM_CONFIG.dims()  # [128, 64, 48] for latent space
        """
        latent_sample = ofm_pipeline.sample_with_odeint(
            dims=latent_dims,
            conds=conditions,
            n_channels=OFM_CONFIG.n_chan,  # OFM generates 1-channel latent representations
            n_samples=1,
            n_eval=50,
            method='dopri5'
        )
        """
        latent_sample = ofm_pipeline.sample(
            dims=latent_dims,
            conds=conditions_batch,
            n_channels=OFM_CONFIG.n_chan,  # OFM generates 1-channel latent representations
            n_samples=n_samples,
            n_eval=50,
            method='euler' # heun
        )

        latent_sample = latent_sample / OFM_CONFIG.scale

    print(f"Generated latent sample shape: {latent_sample.shape}")

    # Step 2: Decode latent to high-resolution using AENO
    print("Step 2: Decoding latent to high-resolution with AENO...")
    with torch.no_grad():
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        aeno_batches = []
        for start in range(0, latent_sample.shape[0], batch_size):
            batch = latent_sample[start:start + batch_size]
            # AENO expects input of shape [batch, H, W, T] but we have [batch, channels, H, W, T]
            # Need to adjust dimensions - AENO might expect different input format
            # For now, assume latent_sample is already in the right format for AENO
            aeno_out = aeno_model(batch, decode=True)

            # take the average of the log10 scale 
            aeno_out[:, 3] = torch.mean(aeno_out[:, 3], dim=(1, 2, 3), keepdim=True)

            # aeno_out[:, :3] = apply_lowpass_filter_to_tensor(
            #     aeno_out[:, :3],
            #     cutoff_hz=LOW_RATE_CUTOFF_HZ,
            #     fs_hz=FS_HZ//2,
            #     mask_order=FFT_MASK_ORDER,
            #     pad_samples=FFT_PAD_SAMPLES,
            # )
            aeno_batches.append(aeno_out)
        high_res_from_aeno = torch.cat(aeno_batches, dim=0)

    print(f"AENO output shape: {high_res_from_aeno.shape}")
    print(f'prediction amplitude AENO : {high_res_from_aeno[:,3].mean().item()}')

    # Step 3: Super-resolve using SNO
    print("Step 3: Super-resolving with SNO...")
    sno_batch = batch_size // 2
    with torch.no_grad():
        sno_batches = []
        for start in range(0, high_res_from_aeno.shape[0], sno_batch):
            batch = high_res_from_aeno[start:start + sno_batch]
            # SNO expects low-res input, so we need to downsample high_res_from_aeno
            # For now, assume high_res_from_aeno is already suitable as low-res input for SNO
            # In practice, you might need to apply proper downsampling
            sno_out = sno_model(batch)
            sno_batches.append(sno_out)
        final_output = torch.cat(sno_batches, dim=0)
    print(f"SNO output shape: {final_output.shape}")

    print(f'prediction amplitude SNO : {final_output[:,3].mean().item()}')
    # Step 4: Load ground truth and compute metrics
    print("Step 4: Loading ground truth and computing metrics...")

    # Load ground truth high-resolution data for this sample
    if sample_idx >= len(high_res_dataset):
        raise ValueError(f"Sample index {sample_idx} is out of range for ground truth data")

    ground_truth = high_res_dataset[sample_idx].unsqueeze(0).to(SNO_CONFIG.device)
    print(f"Loaded ground truth shape: {ground_truth.shape}")
    print(f'ground truth amplitude : {ground_truth[:,3].mean().item()}')

    # denormalize the ground truth, for plot only 
    ground_truth = denormalize_displacement(ground_truth)
    final_output = denormalize_displacement(final_output)
    selected_mw_row = select_bias_row_from_conditions(conditions)
    final_output = apply_frequency_bias_correction(
        final_output,
        bias_curve_path=BIAS_CURVE_PATH,
        sampling_freq=4.0,
        mw_row=selected_mw_row,
    )
    print(f"Applied frequency-domain bias correction using row {selected_mw_row} from {BIAS_CURVE_PATH}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    # final_output = apply_lowpass_filter_to_tensor(
    #     final_output,
    #     cutoff_hz=FULL_RATE_CUTOFF_HZ,
    #     fs_hz=FS_HZ,
    #     mask_order=FFT_MASK_ORDER,
    #     pad_samples=FFT_PAD_SAMPLES,
    # )

    # Compute metrics comparing final output to ground truth
    with torch.no_grad():
        mse_loss = torch.mean((final_output - ground_truth)**2).item()
        mae_loss = torch.mean(torch.abs(final_output - ground_truth)).item()

    print(f"MSE vs ground truth: {mse_loss:.6f}")
    print(f"MAE vs ground truth: {mae_loss:.6f}")

    # Move corrected outputs off GPU and release device memory early.
    final_output_cpu_all = final_output.detach().cpu()
    ground_truth_cpu_all = ground_truth.detach().cpu()
    del final_output
    del ground_truth
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    output_dir = None
    save_dir = None
    if save_outputs or not skip_plots:
        output_dir = TEMP_PLOT_ROOT / f"E2E_test_results_{post_fix}"
        output_dir.mkdir(parents=True, exist_ok=True)

        save_dir = SAVE_PLOT_ROOT / "M7_scen"
        save_dir.mkdir(parents=True, exist_ok=True)

    # Save the intermediate and final results
    results = {
        'sample_idx': sample_idx,
        'latent_shape': tuple(latent_sample.shape),
        'aeno_output_shape': tuple(high_res_from_aeno.shape),
        'final_output_shape': tuple(final_output_cpu_all.shape),
        'ground_truth_shape': tuple(ground_truth_cpu_all.shape),
        'mse_vs_ground_truth': mse_loss,
        'mae_vs_ground_truth': mae_loss,
        'conditions': conditions.cpu().numpy(),
    }

    if not skip_plots:
        # Create comparison visualization similar to sno_test
        # Convert tensors to numpy for plotting
        final_cpu = final_output_cpu_all[:1]
        ground_truth_cpu = ground_truth_cpu_all

        # Create spatial comparison plot (final output vs ground truth)
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle(f'End-to-End Results vs Ground Truth - Sample {sample_idx}\nMSE: {mse_loss:.6f}, MAE: {mae_loss:.6f}', fontsize=14)

        # Select 4 time slices: beginning, quarter, middle, three-quarters
        time_indices = [0, 24, 48, 72]  # For time dimension

        for i, t_idx in enumerate(time_indices):
            # Rows: ground truth, final output, error

            # Ground truth (first channel)
            if ground_truth_cpu.shape[1] > 0:
                gt_slice = ground_truth_cpu[0, 0, :, :, t_idx]
                im1 = axes[0, i].imshow(gt_slice, cmap='RdBu_r', aspect='equal')
                axes[0, i].set_title(f'Ground Truth (t={t_idx})')
                axes[0, i].axis('off')
                plt.colorbar(im1, ax=axes[0, i], fraction=0.046)

            # Final E2E output (first channel)
            if final_cpu.shape[1] > 0:
                final_slice = final_cpu[0, 0, :, :, t_idx]
                im2 = axes[1, i].imshow(final_slice, cmap='RdBu_r', aspect='equal')
                axes[1, i].set_title(f'E2E Output (t={t_idx})')
                axes[1, i].axis('off')
                plt.colorbar(im2, ax=axes[1, i], fraction=0.046)

            # Error (absolute difference)
            if final_cpu.shape[1] > 0 and ground_truth_cpu.shape[1] > 0:
                error_slice = torch.abs(final_cpu[0, 0, :, :, t_idx] - ground_truth_cpu[0, 0, :, :, t_idx])
                im3 = axes[2, i].imshow(error_slice, cmap='RdBu_r', aspect='equal')
                axes[2, i].set_title(f'Error (t={t_idx})')
                axes[2, i].axis('off')
                plt.colorbar(im3, ax=axes[2, i], fraction=0.046)

        plt.tight_layout()
        plt.savefig(output_dir / f'e2e_sample_{sample_idx}_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Create waveform comparison for selected stations (E2E output vs ground truth)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'End-to-End Output vs Ground Truth Waveforms - Sample {sample_idx}', fontsize=16)

        # Select representative stations (x, y coordinates)
        stations = [
            (64, 32, "Center"),
            (128, 64, "Mid-Right"),
            (32, 16, "Top-Left"),
            (192, 96, "Bottom-Right"),
            (160, 48, "Right"),
            (96, 80, "Bottom")
        ]

        station_waveforms: List[Tuple[str, np.ndarray, np.ndarray]] = []
        sampling_freq = 4.0  # Hz
        duration = 24.0  # seconds

        for idx, (x, y, label) in enumerate(stations):
            row, col = idx // 3, idx % 3

            # Extract time series at this station for ground truth and E2E output
            try:
                gt_waveform = ground_truth_cpu[0, 0, min(x, ground_truth_cpu.shape[2]-1), min(y, ground_truth_cpu.shape[3]-1), :]  # Ground truth
                e2e_waveform = final_cpu[0, 0, min(x, final_cpu.shape[2]-1), min(y, final_cpu.shape[3]-1), :]  # E2E output

                gt_waveform_np = gt_waveform.numpy()
                e2e_waveform_np = e2e_waveform.numpy()

                # Plot both waveforms
                time_steps = range(len(gt_waveform_np))
                axes[row, col].plot(time_steps, gt_waveform_np, 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
                axes[row, col].plot(time_steps, e2e_waveform_np, 'r--', linewidth=2, label='E2E Output', alpha=0.8)

                axes[row, col].set_title(f'Station {label} (x={x}, y={y})')
                axes[row, col].set_xlabel('Time Step')
                axes[row, col].set_ylabel('Amplitude')
                axes[row, col].grid(True, alpha=0.3)
                axes[row, col].legend()

                # Calculate and display RMSE for this station
                rmse = torch.sqrt(torch.mean((gt_waveform - e2e_waveform)**2)).item()
                axes[row, col].text(0.02, 0.98, f'RMSE: {rmse:.4f}',
                                  transform=axes[row, col].transAxes, fontsize=10,
                                  verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                station_waveforms.append((label, gt_waveform_np, e2e_waveform_np))

            except Exception as e:
                print(f"Warning: Could not plot station {label}: {e}")
                axes[row, col].text(0.5, 0.5, f'Error plotting\nstation {label}',
                                  transform=axes[row, col].transAxes, ha='center', va='center')

        plt.tight_layout()
        plt.savefig(output_dir / f'e2e_sample_{sample_idx}_waveforms.png', dpi=150, bbox_inches='tight')
        plt.close()

        if station_waveforms:
            fig_freq, axes_freq = plt.subplots(2, 3, figsize=(18, 10))
            fig_freq.suptitle(f'End-to-End Frequency Comparison - Sample {sample_idx}', fontsize=16)

            for idx, (label, gt_np, e2e_np) in enumerate(station_waveforms):
                row, col = idx // 3, idx % 3
                n_samples = len(gt_np)

                freqs = np.fft.rfftfreq(n_samples, d=1.0 / sampling_freq)
                gt_fft = np.fft.rfft(gt_np)
                e2e_fft = np.fft.rfft(e2e_np)

                gt_amp = np.abs(gt_fft) / n_samples
                e2e_amp = np.abs(e2e_fft) / n_samples

                epsilon = 1e-12
                gt_amp = np.clip(gt_amp, epsilon, None)
                e2e_amp = np.clip(e2e_amp, epsilon, None)

                if len(freqs) > 1:
                    freqs_plot = freqs[1:]
                    gt_amp_plot = gt_amp[1:]
                    e2e_amp_plot = e2e_amp[1:]
                else:
                    freqs_plot = freqs
                    gt_amp_plot = gt_amp
                    e2e_amp_plot = e2e_amp

                axes_freq[row, col].loglog(freqs_plot, gt_amp_plot, 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
                axes_freq[row, col].loglog(freqs_plot, e2e_amp_plot, 'r--', linewidth=2, label='E2E Output', alpha=0.8)

                axes_freq[row, col].set_title(f'Station {label}')
                axes_freq[row, col].set_xlabel('Frequency (Hz)')
                axes_freq[row, col].set_ylabel('Amplitude')
                axes_freq[row, col].grid(True, which='both', linestyle='--', alpha=0.3)
                axes_freq[row, col].legend()

            plt.tight_layout()
            plt.savefig(output_dir / f'e2e_sample_{sample_idx}_frequency.png', dpi=150, bbox_inches='tight')
            plt.close()

    if save_outputs:
        # Save results
        #results_path = output_dir / f'e2e_sample_{sample_idx}_results.npy'
        #np.save(results_path, results)

        synthetic_path = save_dir / f"e2e_sample_{sample_idx}_synthetic_100.npy"
        ground_truth_path = save_dir / f"e2e_sample_{sample_idx}_ground_truth.npy"
        np.save(synthetic_path, final_output_cpu_all.numpy().astype(np.float32, copy=False))
        np.save(ground_truth_path, ground_truth_cpu_all.numpy().astype(np.float32, copy=False))

        print(f"End-to-end test results saved to {output_dir}")
        print(f"Saved synthetic wavefields to {synthetic_path}")
        print(f"Saved ground truth wavefield to {ground_truth_path}")

    if return_data:
        return results, final_output_cpu_all.numpy(), ground_truth_cpu_all.numpy()

    return results



def build_residual_hdf5(
    h5_path: Path = SAVE_PLOT_ROOT / "residual_data_vel_075_unbias.h5",
    n_events: int = 300,#200,
    stride: int = 1,
    n_samples: int = 100,
    batch_size: int = 8,
    time_step: float = 0.25,
    osc_damping: float = 0.05,
    T: int = 96,
) -> None:

    # only store 5 examples for testing
    start_index = 0
    sample_indices = list(range(start_index, start_index + n_events * stride, stride))
    if not sample_indices:
        raise ValueError("No sample indices to process.")

    print(f"Processing {len(sample_indices)} events: {sample_indices[0]}..{sample_indices[-1]}")

    # Run one event to establish shapes
    freq = np.fft.rfftfreq(T, d=time_step)
    h5_path = Path(h5_path)
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "w") as f:
        validation = f.create_group("validation")
        validation.create_group("FS")
        synthetic = f.create_group("synthetic")
        synthetic.create_group("FS")

        validation.create_dataset("FS/freq", data=freq.astype(np.float32))
        synthetic.create_dataset("FS/freq", data=freq.astype(np.float32))

        val_h_fs = []
        val_v_fs = []
        syn_h_fs = []
        syn_v_fs = []

        for event_idx, sample_idx in enumerate(sample_indices):
            print(f"Event {event_idx + 1}/{len(sample_indices)}: sample_idx={sample_idx}")
            with torch.no_grad():
                _, syn_np, gt_np = end_to_end_test(
                    sample_idx,
                    n_samples=n_samples,
                    batch_size=batch_size,
                    skip_plots=True,
                    save_outputs=False,
                    return_data=True,
                )

            gt = gt_np[0]
            #gt_acc = np.diff(gt, axis=-1, prepend=0.0)
            gt_acc = gt # keep velocity
            gt_acc_flat = gt_acc.reshape(3, -1, T)
            gt_rfft = np.fft.rfft(gt_acc_flat, axis=-1)
            h_fs = np.sqrt((np.abs(gt_rfft[0]) ** 2 + np.abs(gt_rfft[1]) ** 2) / 2)
            v_fs = np.abs(gt_rfft[2])

            val_h_fs.append(h_fs.astype(np.float32))
            val_v_fs.append(v_fs.astype(np.float32))
            fake_h_f_all = []
            fake_v_f_all = []

            for i in range(n_samples):
                #syn_acc = np.diff(syn_np[i], axis=-1, prepend=0.0)
                syn_acc = syn_np[i] # keep velocity
                syn_acc_flat = syn_acc.reshape(3, -1, T).transpose(1, 0, 2)
                wfs = torch.tensor(
                    syn_acc_flat, dtype=torch.float32, device=ANALYSIS_DEVICE
                )

                wfs_f = torch.abs(torch.fft.rfft(wfs, dim=-1)) + 1e-6
                power_mean_f = torch.sqrt((wfs_f[:, 0] ** 2 + wfs_f[:, 1] ** 2) / 2)
                fake_h_f_all.append(power_mean_f.cpu())
                fake_v_f_all.append(torch.abs(wfs_f[:, 2]).cpu())
                del wfs, wfs_f, power_mean_f

            h_f_all = torch.stack(fake_h_f_all, dim=0)
            v_f_all = torch.stack(fake_v_f_all, dim=0)

            h_f_mean, h_f_std = cal_gmean_gstd(h_f_all)
            v_f_mean, v_f_std = cal_gmean_gstd(v_f_all)

            syn_h_fs.append(np.stack([h_f_mean.numpy(), h_f_std.numpy()], axis=0))
            syn_v_fs.append(np.stack([v_f_mean.numpy(), v_f_std.numpy()], axis=0))
            torch.cuda.empty_cache()

        validation.create_dataset("FS/H_FS", data=np.stack(val_h_fs, axis=0))
        validation.create_dataset("FS/V_FS", data=np.stack(val_v_fs, axis=0))

        synthetic.create_dataset("FS/H_FS", data=np.stack(syn_h_fs, axis=0))
        synthetic.create_dataset("FS/V_FS", data=np.stack(syn_v_fs, axis=0))

    print(f"Saved residual data to {h5_path}")


# and  generate 100 synthetic given the same conditions
# that is synthetic of shape [100, 3, 256, 128, 96], 


if __name__ == "__main__":
    try:
        build_residual_hdf5()
        print("Residual dataset creation completed successfully!")
    except Exception as e:
        print(f"Error during residual dataset creation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)