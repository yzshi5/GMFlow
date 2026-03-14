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

# Ensure repo-local imports work when executing from `exp/`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


from utils.autoencoding_operator import AutoEncoderOperator
from utils.latent_ofm_clean_pred import OFMModel
from utils.super_resolution_operator import SuperResolutionOperator
from utils.unet_ofm import UNet_cond


import imageio
from tqdm import tqdm

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

device = "cuda:1" if torch.cuda.is_available() else "cpu"
#post_fix = "x_pred_unif_filter_05_t48_double" # 1c_add, 1c_add_tr
post_fix = "x_pred_unif_filter_075_t48_128M_res_ind_test_query_scale" # 1c_add, 1c_add_tr
# Frequency-domain filtering defaults

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
    scale: float = 1.0  # scale the latent data by this factor
    device: str = device
    t_eps: float = 0.02  # try 0.02
    checkpoint_path: Path = MODEL_ROOT / "OFM" / "epoch_300.pt"
    latent_data_file: Path = DATA_ROOT / "latent_data" /  "mid_075_128_rupture_filter_res_t48.npy"
    model_config = {
        "hidden_channels": 96,
        "num_res_blocks": 2,
        "num_heads": 8,
        "attention_res": "8,4",
        "channel_mult": (1, 2, 3, 4),
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
hypocenters = load_hypocenters(HYPOCENTER_PATH)

#%%
# Load pretrained models.
sno_model = load_sno_model()
aeno_model = load_aeno_model()
ofm_pipeline = load_ofm_model()

#%%


def end_to_end_test(sample_idx: int) -> Dict[str, Any]:
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
    
    # test
    print(f"conditions: {conditions}\n")
    print(f"Using conditions shape: {conditions.shape}")

    # Step 1: Generate latent sample using OFM
    print("Step 1: Generating latent sample with OFM...")
    with torch.no_grad():
        latent_dims = OFM_CONFIG.dims()  # [128, 64, 48] for latent space
        latent_sample = ofm_pipeline.sample(
            dims=latent_dims,
            conds=conditions,
            n_channels=OFM_CONFIG.n_chan,  # OFM generates 1-channel latent representations
            n_samples=1,
            n_eval=50,
            method='euler' # heun
        )

        latent_sample = latent_sample / OFM_CONFIG.scale

    print(f"Generated latent sample shape: {latent_sample.shape}")

    # Step 2: Decode latent to high-resolution using AENO
    print("Step 2: Decoding latent to high-resolution with AENO...")
    with torch.no_grad():
        # AENO expects input of shape [batch, H, W, T] but we have [batch, channels, H, W, T]
        # Need to adjust dimensions - AENO might expect different input format
        # For now, assume latent_sample is already in the right format for AENO
        high_res_from_aeno = aeno_model(latent_sample, decode=True)

        # take the average of the log10 scale 
        high_res_from_aeno[:,3] = torch.mean(high_res_from_aeno[:,3], dim=(1, 2, 3), keepdim=True)



    print(f"AENO output shape: {high_res_from_aeno.shape}")
    print(f'prediction amplitude AENO : {high_res_from_aeno[:,3].mean().item()}')

    # Step 3: Super-resolve using SNO
    print("Step 3: Super-resolving with SNO...")
    with torch.no_grad():
        # SNO expects low-res input, so we need to downsample high_res_from_aeno
        # For now, assume high_res_from_aeno is already suitable as low-res input for SNO
        # In practice, you might need to apply proper downsampling
        final_output_q1 = sno_model(high_res_from_aeno, query_scale=1)
        final_output_q2 = sno_model(high_res_from_aeno, query_scale=2)
    print(f"SNO output shape (query_scale=1): {final_output_q1.shape}")
    print(f"SNO output shape (query_scale=2): {final_output_q2.shape}")

    print(f'prediction amplitude SNO (query_scale=1): {final_output_q1[:,3].mean().item()}')
    print(f'prediction amplitude SNO (query_scale=2): {final_output_q2[:,3].mean().item()}')
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
    final_output = denormalize_displacement(final_output_q1)
    final_output_q2 = denormalize_displacement(final_output_q2)



    # Compute metrics comparing final output to ground truth
    with torch.no_grad():
        mse_loss = torch.mean((final_output - ground_truth)**2).item()
        mae_loss = torch.mean(torch.abs(final_output - ground_truth)).item()

    print(f"MSE vs ground truth: {mse_loss:.6f}")
    print(f"MAE vs ground truth: {mae_loss:.6f}")

    # Create output directory
    output_dir = TEMP_PLOT_ROOT / f"E2E_test_results_{post_fix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save denormalized tensors to .npy files in final_plot/save_plot/supres
    npy_output_dir = SAVE_PLOT_ROOT / "supres"
    npy_output_dir.mkdir(parents=True, exist_ok=True)
    np.save(npy_output_dir / f"ground_truth_sample_{sample_idx}.npy", ground_truth.detach().cpu().numpy())
    np.save(npy_output_dir / f"sno_query_scale_1_denorm_sample_{sample_idx}.npy", final_output.detach().cpu().numpy())
    np.save(npy_output_dir / f"sno_query_scale_2_denorm_sample_{sample_idx}.npy", final_output_q2.detach().cpu().numpy())
    print(f"Saved denormalized .npy files for sample {sample_idx} to {npy_output_dir}")

    def save_single_video(tensor: torch.Tensor, tag: str) -> None:
        video_tensor = tensor.detach().cpu()
        if video_tensor.shape[0] == 1:
            video_tensor = video_tensor[0]
        video_data = video_tensor[:3].permute(1, 2, 0, 3).numpy()
        video_path = output_dir / f"{tag}_sample_{sample_idx}_magnitude.mp4"
        create_seismic_video_frames(video_data, component='magnitude', save_path=str(video_path), fps=8)

    print("Saving single-field videos...")
    try:
        save_single_video(ground_truth, "ground_truth")
        save_single_video(final_output, "sno_query_scale_1")
        save_single_video(final_output_q2, "sno_query_scale_2")
        print("Single-field videos saved successfully!")
    except Exception as e:
        print(f"Warning: Failed to save single-field videos: {e}")

    # Save the intermediate and final results


    # Create comparison visualization for end-to-end output
    # Convert tensors to numpy for plotting
    final_cpu = final_output.cpu()
    ground_truth_cpu = ground_truth.cpu()

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

    # Create comparison video (E2E output vs ground truth)
    print("Creating comparison video...")
    try:
        create_comparison_video_from_tensors(ground_truth,final_output,
                                           sample_idx, 'magnitude', 8, output_dir, "e2e_vs_gt")
        print("Comparison video created successfully!")
    except Exception as e:
        print(f"Warning: Failed to create comparison video: {e}")

    # Save results
    print(f"End-to-end test results saved to {output_dir}")



def create_seismic_video_frames(data, component='magnitude', save_path='seismic_animation.mp4',
                               fps=4, dpi=100, figsize=(8, 8)):
    """
    Create a video from seismic data using imageio (no ffmpeg required)

    Parameters:
    - data: numpy array of shape [H, W, 3, T] where 3 represents X, Y, Z components
    - component: which component to visualize ('X', 'Y', 'Z', 'magnitude', 'horizontal')
    - save_path: path to save the video
    - fps: frames per second
    - dpi: dots per inch for the figure
    - figsize: figure size (width, height)
    """

    H, W, _, T = data.shape
    dt = 0.25  # time step in seconds

    # Prepare data based on component
    if component == 'X':
        plot_data = data[:, :, 0, :]  # X component
    elif component == 'Y':
        plot_data = data[:, :, 1, :]  # Y component
    elif component == 'Z':
        plot_data = data[:, :, 2, :]  # Z component
    elif component == 'magnitude':
        plot_data = np.sqrt(data[:, :, 0, :]**2 + data[:, :, 1, :]**2 + data[:, :, 2, :]**2)
    elif component == 'horizontal':
        plot_data = np.sqrt(data[:, :, 0, :]**2 + data[:, :, 1, :]**2)
    else:
        raise ValueError("component must be 'X', 'Y', 'Z', 'magnitude', or 'horizontal'")

    # Calculate global min/max for consistent colorbar
    vmin = np.percentile(plot_data, 1)  # Use 1st percentile to avoid outliers
    vmax = np.percentile(plot_data, 99)  # Use 99th percentile to avoid outliers

    print(f"Creating {T} frames for {component} component...")

    # Create frames
    frames = []
    for frame in tqdm(range(T), desc=f"Creating {component} frames"):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0.06, 0.08, 0.84, 0.84])

        # Preserve aspect ratio while centering the image
        im = ax.imshow(plot_data[:, :, frame], cmap='RdBu_r', vmin=vmin, vmax=vmax,
                       aspect='equal', origin='lower')
        ax.set_anchor('C')

        # Add colorbar in a fixed side axis (avoid shifting the main plot)
        cax = fig.add_axes([0.92, 0.08, 0.02, 0.84])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(f'{component} Component Amplitude', rotation=270, labelpad=20)

        # Set labels and title
        ax.set_xlabel('Width (stations)')
        ax.set_ylabel('Height (stations)')
        current_time = frame * dt
        ax.set_title(f'Seismic Wave Propagation - {component} Component\nTime: {current_time:.2f}s')

        # Add time text
        ax.text(0.02, 0.98, f'Frame: {frame+1}/{T}\nTime: {current_time:.2f}s',
                transform=ax.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Convert to image
        fig.canvas.draw()
        # Get the buffer and convert to numpy array
        buf = fig.canvas.buffer_rgba()
        frame_array = np.asarray(buf)
        # Convert RGBA to RGB
        frame_array = frame_array[:, :, :3]
        frames.append(frame_array)

        plt.close(fig)

    # Save as video using imageio
    print(f"Saving video to {save_path}...")
    imageio.mimsave(save_path, frames, fps=fps)
    print(f"Video saved successfully to {save_path}")


def create_comparison_video_from_tensors(gt_tensor: torch.Tensor, recon_tensor: torch.Tensor,
                                        sample_index: int, component: str = 'magnitude',
                                        fps: int = 4, output_dir: Path = None, prefix: str = "comparison"):
    """
    Create a side-by-side comparison video from ground truth and reconstructed tensors.
    Uses all available time frames for the animation.

    Args:
        gt_tensor (torch.Tensor): Ground truth tensor [batch, channels, H, W, T]
        recon_tensor (torch.Tensor): Reconstructed tensor [batch, channels, H, W, T]
        sample_index (int): Index of the sample for naming
        component (str): Which component to visualize
        fps (int): Frames per second
        output_dir (Path): Output directory
        prefix (str): Prefix for filenames
    """
    if output_dir is None:
        output_dir = TEMP_PLOT_ROOT / f"{prefix}_results_{post_fix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to numpy for video creation (permute to [H, W, C, T] format)
    print(f"Converting tensors: gt.shape={gt_tensor.shape}, recon.shape={recon_tensor.shape}")
    numpy_gt = gt_tensor.cpu().numpy()
    numpy_recon = recon_tensor.cpu().numpy()

    # Handle batch dimension
    if numpy_gt.shape[0] == 1:
        numpy_gt = numpy_gt[0]  # [channels, H, W, T]
        numpy_recon = numpy_recon[0]

    # Convert to [H, W, C, T] format
    gt_data = numpy_gt.transpose(1, 2, 0, 3)  # [H, W, C, T]
    recon_data = numpy_recon.transpose(1, 2, 0, 3)  # [H, W, C, T]

    print(f"Video data shapes: gt={gt_data.shape}, recon={recon_data.shape}")
    print(f"Using all {gt_data.shape[-1]} time frames for video")

    # Create comparison animation
    mp4_path = output_dir / f'{prefix}_sample_{sample_index}_{component}_all_frames.mp4'
    gif_path = output_dir / f'{prefix}_sample_{sample_index}_{component}_all_frames.gif'

    try:
        create_side_by_side_video(
            gt_data, recon_data, component=component,
            save_path=str(mp4_path), fps=fps
        )
        print(f"MP4 video saved: {mp4_path}")
    except Exception as e:
        print(f"MP4 creation failed ({e}), trying GIF format...")
        try:
            create_side_by_side_video(
                gt_data, recon_data, component=component,
                save_path=str(gif_path), fps=min(fps, 10)
            )
            print(f"GIF animation saved: {gif_path}")
        except Exception as e2:
            print(f"GIF creation failed: {e2}")

    return gt_data, recon_data


def create_side_by_side_video(gt_data, recon_data, component='magnitude',
                            save_path='sno_comparison.mp4', fps=4, figsize=(12, 6)):
    """
    Create a side-by-side comparison video showing ground truth vs reconstructed data.

    Args:
        gt_data: Ground truth data [H, W, C, T]
        recon_data: Reconstructed data [H, W, C, T]
        component: Which component to visualize
        save_path: Path to save the video
        fps: Frames per second
        figsize: Figure size for the video
    """
    _, _, _, T = gt_data.shape
    dt = 0.25  # time step in seconds (total simulation time = T * dt)

    print(f"Video will show all {T} time frames (total simulation time: {T * dt:.2f} seconds)")
    print(f"At {fps} fps, video duration will be {T / fps:.1f} seconds")

    # Prepare data based on component
    if component == 'X':
        gt_plot = gt_data[:, :, 0, :]
        recon_plot = recon_data[:, :, 0, :]
    elif component == 'Y':
        gt_plot = gt_data[:, :, 1, :]
        recon_plot = recon_data[:, :, 1, :]
    elif component == 'Z':
        gt_plot = gt_data[:, :, 2, :]
        recon_plot = recon_data[:, :, 2, :]
    elif component == 'magnitude':
        gt_plot = np.sqrt(gt_data[:, :, 0, :]**2 + gt_data[:, :, 1, :]**2 + gt_data[:, :, 2, :]**2)
        recon_plot = np.sqrt(recon_data[:, :, 0, :]**2 + recon_data[:, :, 1, :]**2 + recon_data[:, :, 2, :]**2)
    elif component == 'horizontal':
        gt_plot = np.sqrt(gt_data[:, :, 0, :]**2 + gt_data[:, :, 1, :]**2)
        recon_plot = np.sqrt(recon_data[:, :, 0, :]**2 + recon_data[:, :, 1, :]**2)
    else:
        raise ValueError("component must be 'X', 'Y', 'Z', 'magnitude', or 'horizontal'")

    # Calculate global min/max for consistent colorbar across both plots
    all_data = np.concatenate([gt_plot.flatten(), recon_plot.flatten()])
    vmin = np.percentile(all_data, 1)
    vmax = np.percentile(all_data, 99)

    print(f"Creating {T} comparison frames for {component} component...")

    frames = []
    for frame in tqdm(range(T), desc=f"Creating {component} comparison frames"):
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.05)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        cax = fig.add_subplot(gs[0, 2])

        # Ground truth plot
        im1 = ax1.imshow(gt_plot[:, :, frame], cmap='RdBu_r', vmin=vmin, vmax=vmax,
                        aspect='equal', origin='lower')
        ax1.set_anchor('C')
        ax1.set_title('Ground Truth')
        ax1.set_xlabel('Width (stations)')
        ax1.set_ylabel('Height (stations)')

        # Reconstructed plot
        im2 = ax2.imshow(recon_plot[:, :, frame], cmap='RdBu_r', vmin=vmin, vmax=vmax,
                        aspect='equal', origin='lower')
        ax2.set_anchor('C')
        ax2.set_title('SNO Reconstructed')
        ax2.set_xlabel('Width (stations)')
        ax2.set_ylabel('Height (stations)')

        # Add shared colorbar without shifting the main plots
        cbar = fig.colorbar(im2, cax=cax)
        cbar.set_label(f'{component} Component Amplitude', rotation=270, labelpad=20)

        # Overall title with time
        current_time = (frame + 1) * dt
        fig.suptitle(f'Waveform Comparison - {component} Component\nTime: {current_time:.2f}s (Frame {frame+1}/{T})',
                    fontsize=14)

        # Convert to image
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        frame_array = np.asarray(buf)
        frame_array = frame_array[:, :, :3]
        frames.append(frame_array)

        plt.close(fig)

    # Save as video
    print(f"Saving comparison video to {save_path}...")
    imageio.mimsave(save_path, frames, fps=fps)
    print(f"Comparison video saved successfully to {save_path}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"

    if mode == "run":
        # Print configuration and dataset information
        print("SNO configuration:", SNO_CONFIG)
        print("AENO configuration:", AENO_CONFIG)
        print("OFM configuration:", OFM_CONFIG)
        print(f"High-resolution samples: {len(high_res_dataset)}")
        print(f"Hypocenter tensor shape: {hypocenters.shape}")
        print("Loaded SNO model to device:", SNO_CONFIG.device)
        print("Loaded AENO model to device:", AENO_CONFIG.device)
        print("Loaded OFM network to device:", OFM_CONFIG.device)

    elif mode == "end2end_test":
        sample_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        try:
            end_to_end_test(sample_idx)
            print("End-to-end test completed successfully!")
        except Exception as e:
            print(f"Error during end-to-end test: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    else:
        print("Usage:")
        print("  python end_to_end_eval.py run")
        print("  python end_to_end_eval.py end2end_test [sample_idx]")
        print("    sample_idx: sample index (default: 0)")
        sys.exit(1)

