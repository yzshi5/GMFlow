"""
Utility for producing final M6 "clean" datasets and their normalized copies.

Pipeline:
  1. Load every file in processed_M6/m6_256_128 (shape [3, 256, 128, T]).
  2. Apply a 1 Hz FFT low-pass filter (fs=4 Hz) and save the filtered volume
     without changing its resolution to final_norm/m6_256_128_clean.
  3. Apply a 0.75 Hz low-pass filter to the same input, downsample to
     [3, 128, 64, 48], and save the volumes to
     final_norm/m6_128_64_48_fmax_075_clean.
  4. Normalize all newly created "clean" files with the method used in
     norm_all_M6_M7_M44.py (divide by std, append log10(std) channel) and
     store the outputs under final_norm/normalized.

Usage:
    python generate_final_norm.py            # runs filter + normalize
    python generate_final_norm.py filter     # only produce clean files
    python generate_final_norm.py normalize  # only (re)normalize clean files
"""

#%%
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Sequence
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


#%%
FS_HZ = 4.0
FULL_RATE_CUTOFF_HZ = 1.0
LOW_RATE_CUTOFF_HZ = 0.75
FFT_MASK_ORDER = 6
FFT_PAD_SAMPLES = 8


BASE_PROCESSED = Path(os.getenv("GMFLOW_PROCESSED_ROOT", "path/to/your/processed_output")) / "processed_M6"
INPUT_DIR = BASE_PROCESSED / "m6_256_128"
FINAL_NORM_BASE = BASE_PROCESSED / "final_norm"
CLEAN_FULL_DIR = FINAL_NORM_BASE / "m6_256_128_clean"
CLEAN_DOWNSAMPLED_48_DIR = FINAL_NORM_BASE / "m6_128_64_48_fmax_075_clean"
NORMALIZED_BASE = FINAL_NORM_BASE / "normalized"
NORM_FULL_DIR = NORMALIZED_BASE / "norm_m6_256_128_clean"
NORM_DOWNSAMPLED_48_DIR = NORMALIZED_BASE / "norm_m6_128_64_48_fmax_075_clean"
TARGET_SPATIAL = (128, 64)
TARGET_TIME_LENGTH = 48


def _numeric_key(path: Path) -> tuple[int, str]:
    """Sort helper that extracts the integer after '_sim_' if present."""
    stem = path.stem
    if "_sim_" in stem:
        candidate = stem.split("_sim_")[-1]
        digits = "".join(ch for ch in candidate if ch.isdigit())
        if digits:
            return (int(digits), path.name)
    return (10**9, path.name)


def _list_npy_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    files = [
        p for p in directory.iterdir()
        if p.suffix == ".npy" and p.is_file() and not p.name.startswith("._")
    ]
    files.sort(key=_numeric_key)
    return files


def _extract_sim_suffix(path: Path) -> str:
    stem = path.stem
    if "_sim_" in stem:
        suffix = stem.split("_sim_")[-1]
        digits = "".join(ch for ch in suffix if ch.isdigit())
        if digits:
            return digits
        return suffix
    return stem


def freq_to_mask_lowpass(
    corner_freq,
    length,
    sampling_freq=100.0,
    np_lp=4,
):
    """Return a low-pass attenuation mask suitable for rFFT magnitudes."""
    cf = torch.as_tensor(corner_freq, dtype=torch.float32)
    if cf.ndim == 1:
        fc_values = cf
    else:
        fc_values = cf[:, -1]

    num_freq_bins = length // 2 + 1
    freq = torch.linspace(0.0, sampling_freq / 2.0, steps=num_freq_bins)
    freq_mask = torch.zeros((fc_values.shape[0], 1, num_freq_bins), dtype=torch.float32)

    for i, fc_lp in enumerate(fc_values):
        fc_lp = torch.clamp(fc_lp, min=1e-6)
        freq_mask[i, 0] = 1.0 / torch.sqrt(1.0 + (freq / fc_lp) ** (2 * np_lp))

    return freq_mask


def _lowpass_mask(length: int, cutoff_hz: float, fs_hz: float, mask_order: int) -> np.ndarray:
    mask_profile = freq_to_mask_lowpass(
        torch.tensor([[0.0, cutoff_hz]], dtype=torch.float32),
        length=length,
        sampling_freq=fs_hz,
        np_lp=mask_order,
    )[0, 0]
    return mask_profile.cpu().numpy().astype(np.float32, copy=False)


def apply_fft_lowpass_filter(
    data: np.ndarray,
    cutoff_hz: float,
    fs_hz: float,
    mask_order: int = FFT_MASK_ORDER,
    pad_samples: int = FFT_PAD_SAMPLES,
) -> np.ndarray:
    """Frequency-domain low-pass filter along the last axis."""
    if data.ndim != 4:
        raise ValueError(f"Expected [C, X, Y, T], got {data.shape}")

    pad = max(int(pad_samples), 0)
    working = np.asarray(data, dtype=np.float32, order="C")
    if pad:
        working = np.pad(
            working,
            ((0, 0), (0, 0), (0, 0), (pad, pad)),
            mode="constant",
        )

    length = working.shape[-1]
    mask = _lowpass_mask(length, cutoff_hz, fs_hz, mask_order)
    mask_shape = (1,) * (working.ndim - 1) + (mask.shape[0],)
    spectrum = np.fft.rfft(working, axis=-1)
    filtered = np.fft.irfft(spectrum * mask.reshape(mask_shape), n=length, axis=-1)

    if pad:
        filtered = filtered[..., pad:-pad]

    return filtered.astype(np.float32, copy=False)


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


def downsample_time(data: np.ndarray, target_length: int) -> np.ndarray:
    return _downsample_axis(data, target_length, axis=-1).astype(np.float32, copy=False)


def save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array.astype(np.float32, copy=False))


def process_single_file(
    file_path: Path,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Filter one input cube and emit all clean variants."""
    sim_id = _extract_sim_suffix(file_path)
    full_clean_path = CLEAN_FULL_DIR / f"m6_clean_256_128_sim_{sim_id}.npy"
    clean_48_path = (
        CLEAN_DOWNSAMPLED_48_DIR / f"m6_clean_128_64_48_fmax_075_sim_{sim_id}.npy"
    )

    if not overwrite and full_clean_path.exists() and clean_48_path.exists():
        return {
            "full": full_clean_path,
            "clean_48": clean_48_path,
        }

    data = np.load(file_path, allow_pickle=False)
    if data.ndim != 4 or data.shape[0] != 3:
        raise ValueError(f"Unexpected shape {data.shape} for {file_path}")

    filtered_full = apply_fft_lowpass_filter(
        data,
        cutoff_hz=FULL_RATE_CUTOFF_HZ,
        fs_hz=FS_HZ,
    )
    save_array(full_clean_path, filtered_full)

    filtered_low = apply_fft_lowpass_filter(
        data,
        cutoff_hz=LOW_RATE_CUTOFF_HZ,
        fs_hz=FS_HZ,
    )
    spatial_ds = downsample_spatial(filtered_low, TARGET_SPATIAL)
    temporal_48 = downsample_time(spatial_ds, TARGET_TIME_LENGTH)

    save_array(clean_48_path, temporal_48)

    return {
        "full": full_clean_path,
        "clean_48": clean_48_path,
    }


def generate_clean_datasets(
    files: Sequence[Path],
    overwrite: bool = False,
) -> None:
    if not files:
        print(f"No .npy files found in {INPUT_DIR}")
        return

    CLEAN_FULL_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_DOWNSAMPLED_48_DIR.mkdir(parents=True, exist_ok=True)

    for file_path in tqdm(files, desc="Filtering M6 data"):
        try:
            process_single_file(file_path, overwrite=overwrite)
        except Exception as exc:
            print(f"Failed to process {file_path.name}: {exc}")


def normalize_tensor(data: np.ndarray) -> np.ndarray:
    if data.ndim != 4 or data.shape[0] != 3:
        raise ValueError(f"Expected clean tensor with 3 channels, got {data.shape}")
    std_value = float(data.std())
    if not np.isfinite(std_value) or std_value <= 0.0:
        raise ValueError("Standard deviation must be positive for normalization.")
    normalized = data / std_value
    log_std_channel = np.full_like(normalized[:1], np.log10(std_value), dtype=np.float32)
    return np.concatenate([normalized, log_std_channel], axis=0).astype(np.float32, copy=False)


def normalize_directory(
    clean_dir: Path,
    norm_dir: Path,
    overwrite: bool = False,
) -> None:
    clean_files = _list_npy_files(clean_dir)
    if not clean_files:
        print(f"No clean files found in {clean_dir}")
        return

    norm_dir.mkdir(parents=True, exist_ok=True)
    for file_path in tqdm(clean_files, desc=f"Normalizing {clean_dir.name}"):
        output_name = file_path.name.replace("clean", "norm", 1)
        out_path = norm_dir / output_name
        if out_path.exists() and not overwrite:
            continue
        try:
            data = np.load(file_path, allow_pickle=False)
            normalized = normalize_tensor(data)
            save_array(out_path, normalized)
        except Exception as exc:
            print(f"Failed to normalize {file_path.name}: {exc}")


def normalize_clean_outputs(overwrite: bool = False) -> None:
    normalize_directory(CLEAN_FULL_DIR, NORM_FULL_DIR, overwrite=overwrite)
    normalize_directory(CLEAN_DOWNSAMPLED_48_DIR, NORM_DOWNSAMPLED_48_DIR, overwrite=overwrite)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate filtered + normalized M6 datasets.")
    parser.add_argument(
        "mode",
        nargs="?",
        default="all",
        choices=["all", "filter", "normalize"],
        help="Pipeline stage to run.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index within the sorted input list.",
    )
    parser.add_argument(
        "--max-count",
        type=int,
        default=None,
        help="Process at most this many files (filter stage only).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute outputs even if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    files = _list_npy_files(INPUT_DIR)
    if args.mode in {"all", "filter"}:
        start = max(args.start, 0)
        end = len(files) if args.max_count is None else min(len(files), start + max(args.max_count, 0))
        generate_clean_datasets(files[start:end], overwrite=args.overwrite)

    if args.mode in {"all", "normalize"}:
        normalize_clean_outputs(overwrite=args.overwrite)


if __name__ == "__main__":
    main()