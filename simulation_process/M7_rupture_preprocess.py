# Configure roots via GMFLOW_RAW_SIM_ROOT and GMFLOW_PROCESSED_ROOT
#%%
import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os
import sys
import time
import datetime
from pathlib import Path
import matplotlib.animation as animation
from matplotlib.colors import Normalize
import seaborn as sns
from PIL import Image
import imageio
import ast

#%%

# -----------------------------------------------------------------------------
# PATH CONFIG (edit once here; optional env vars override these defaults)
# -----------------------------------------------------------------------------
# Example:
#   export GMFLOW_RAW_SIM_ROOT="/path/to/your/GMFlow_raw_simulation"
#   export GMFLOW_PROCESSED_ROOT="/path/to/your/processed_output"
#
# Raw simulation dataset source:
# https://huggingface.co/datasets/Yaozhong/GMFlow_raw_simulation/tree/main
RAW_SIM_ROOT = Path(os.getenv('GMFLOW_RAW_SIM_ROOT', 'path/to/your/GMFlow_raw_simulation'))
PROCESSED_ROOT = Path(os.getenv('GMFLOW_PROCESSED_ROOT', 'path/to/your/processed_output'))

M7_RESULTS_DIR = RAW_SIM_ROOT / 'final_results_M7_1000'
M7_SUMMARY_CSV = RAW_SIM_ROOT / 'M7_rupture_realization_summary.csv'
M7_PROCESSED_DIR = PROCESSED_ROOT / 'processed_M7'
M7_PROCESSED_RAW_DIR = M7_PROCESSED_DIR / 'raw'
M7_PLOTS_DIR = PROCESSED_ROOT / 'temp_plot' / 'M7'


def _ensure_paths_configured() -> None:
    if 'path/to/your' in str(RAW_SIM_ROOT) or 'path/to/your' in str(PROCESSED_ROOT):
        raise ValueError(
            'Please set RAW_SIM_ROOT/PROCESSED_ROOT in this file or via env vars '
            '(GMFLOW_RAW_SIM_ROOT, GMFLOW_PROCESSED_ROOT) before running.'
        )


def process_rupture_data(start_idx: int = 0, max_count: int | None = None):
    """Process rupture simulation data files and extract hypocenters.

    Parameters:
    - start_idx: global start index in the filtered file list
    - max_count: process at most this many files; if None, process to the end
    """
    _ensure_paths_configured()
    save_path = M7_PROCESSED_RAW_DIR
    save_path_hypo = M7_PROCESSED_DIR
    # Create directories if they don't exist
    save_path.mkdir(parents=True, exist_ok=True)
    save_path_hypo.mkdir(parents=True, exist_ok=True)

    data_path = M7_RESULTS_DIR
    #testdata_path = os.path.join(data_path, os.listdir(data_path)[0])
    #filename = os.path.join(testdata_path, os.listdir(testdata_path)[0])
    file_name_all = os.listdir(data_path)

    # Filter out macOS metadata files (._ files) and other hidden/system files
    file_name_all = [f for f in file_name_all if not f.startswith('._') and not f.startswith('.')]
    # Enforce deterministic order across runs
    file_name_all = sorted(file_name_all)
    print(f"Total valid files: {len(file_name_all)}")

    # Apply batching window
    total_files = len(file_name_all)
    if max_count is not None:
        end_idx = min(start_idx + max_count, total_files)
    else:
        end_idx = total_files
    file_name_all = file_name_all[start_idx:end_idx]
    print(f"Processing files [{start_idx}:{end_idx})")

    # for test

    #file_name_all = file_name_all[:3]
    # load the metadata file
    center_file = M7_SUMMARY_CSV
    center_df = pd.read_csv(center_file)

    ## extract the data and corresponding hypocenter
    hypocenter_all = []
    npts = 118 # dt = 0.25

    for count, file_name in enumerate(tqdm(file_name_all), start=start_idx):
        file_path = os.path.join(data_path, file_name)

        # Skip if output already exists (resumable)
        out_path = save_path / f'm7_raw_r_sim_{count}.npy'
        if out_path.exists():
            continue
        
        # Check if file exists in center_df
        matching_rows = center_df[center_df['Filename'] == file_name]
        assert len(matching_rows) > 0, f"Error: {file_name} not found in center_df"
        
        ## extract the hyocenter
        hypocenter_str = matching_rows['Hypocenter'].iloc[0]

        ## assert hypocenter_str is non-empty
        if pd.isna(hypocenter_str) or len(str(hypocenter_str)) == 0:
            print(f"Warning: Hypocenter is empty for {file_name}, skipping...")
            continue

        # Convert string to numpy array
        # Handle different string formats: "[1, 2, 3]", "(1, 2, 3)", "1, 2, 3", or "1 2 3"
        if isinstance(hypocenter_str, str):
            try:
                # Try parsing as Python literal (list or tuple)
                hypocenter = np.array(ast.literal_eval(hypocenter_str))
            except (ValueError, SyntaxError):
                # If that fails, try comma-separated or space-separated
                # Remove brackets/parentheses if present
                clean_str = hypocenter_str.strip('[]()')
                # Try splitting by comma first
                if ',' in clean_str:
                    hypocenter = np.array([float(x.strip()) for x in clean_str.split(',')])
                else:
                    # Otherwise split by whitespace
                    hypocenter = np.array([float(x) for x in clean_str.split()])
        else:
            # If it's already not a string, convert directly
            hypocenter = np.array(hypocenter_str)

        hypocenter_all.append(hypocenter)

        # extract the data using context manager for proper file handling
        with h5py.File(file_path, 'r') as f:
            sta_list_all = list(f)[:-1] # skip the dt
            # Build mapping from station name to grid (row, col)
            station_to_rc = {}
            for sta in sta_list_all:
                parts = sta.split('_')
                if len(parts) >= 3:
                    row = int(parts[1]) - 1
                    col = int(parts[2]) - 1
                    station_to_rc[sta] = (row, col)

            # Create grid directly in target layout (3, H, W, T) and fill with float32
            grid_data = np.zeros((3, 367, 171, npts), dtype=np.float32)

            for sta in sta_list_all:
                row, col = station_to_rc.get(sta, (None, None))
                if row is None:
                    continue
                x = np.array(f[sta]['X'], dtype=np.float32)
                y = np.array(f[sta]['Y'], dtype=np.float32)
                z = np.array(f[sta]['Z'], dtype=np.float32)
                grid_data[0, row, col, :] = x
                grid_data[1, row, col, :] = y
                grid_data[2, row, col, :] = z

        # Already (3, H, W, T) float32
        data_all_numpy = grid_data

        # subfolder "raw",
        np.save(out_path, data_all_numpy)
        ## save the processed data 

    # Save hypocenters (cast to float32 if numeric)
    hypo_all = np.array(hypocenter_all)
    if np.issubdtype(hypo_all.dtype, np.floating):
        hypo_all = hypo_all.astype(np.float32, copy=False)
    np.save(save_path_hypo / 'm7_hypo_all.npy', hypo_all)


def test_processed_sample(
    processed_dir: Path = M7_PROCESSED_RAW_DIR,
    save_dir: Path = M7_PLOTS_DIR,
    sample_index: int = 0,
    sample_index_list: list[int] | None = [0, 100, 200, 300],
    hypo_path: Path = M7_PROCESSED_DIR / 'm7_hypo_all.npy',
):
    """Load processed .npy files, plot several time slices, and save to temp.

    Parameters:
    - processed_dir: directory containing processed .npy files
    - save_dir: directory to save the slice figure(s)
    - sample_index: single index to visualize if sample_index_list is None
    - time_indices: list of time indices to plot; if None, uses 5 evenly spaced
    - sample_index_list: optional list of indices to visualize sequentially
    - hypo_path: path to saved hypocenter array (m7_hypo_all.npy)
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Gather valid .npy files and exclude macOS metadata (dot-underscore) and hidden files
    files = [
        f for f in os.listdir(processed_dir)
        if f.endswith('.npy') and not f.startswith('._') and not f.startswith('.')
    ]
    # Keep only actual files
    files = [f for f in files if (processed_dir / f).is_file()]
    # Sort by numeric suffix to align with saved order m7_raw_r_sim_{count}.npy
    def _num_key(name: str) -> int:
        base = Path(name).stem
        try:
            if '_sim_' in base:
                return int(base.split('_sim_')[-1])
        except Exception:
            pass
        # Fallback: place unsortable names after numbered ones
        return 10**9
    files.sort(key=_num_key)
    assert len(files) > 0, f"No .npy files found in {processed_dir}"

    # Load hypocenters
    hypo = np.load(hypo_path, allow_pickle=True)
    if isinstance(hypo, np.ndarray) and hypo.dtype == object:
        try:
            hypo = np.array(hypo.tolist())
        except Exception:
            pass
 
    # Determine which indices to render
    indices = sample_index_list if sample_index_list is not None else [sample_index]
    indices = [min(max(idx, 0), len(files) - 1) for idx in indices]

    for idx in indices:
        npy_path = processed_dir / files[idx]

        # Allow loading files saved with object dtype
        data = np.load(npy_path, allow_pickle=True)
        # Unwrap/convert object arrays if present
        if isinstance(data, np.ndarray) and data.dtype == object:
            if data.ndim == 0:
                data = data.item()
            else:
                try:
                    data = np.array(data.tolist())
                except Exception:
                    pass
        assert isinstance(data, np.ndarray), f"Loaded data is not an ndarray: {type(data)}"
        assert data.ndim == 4, f"Unexpected data ndim: {data.ndim}, shape: {data.shape}"
        # Normalize layout to (3, H, W, T)
        if data.shape[0] == 3:
            pass
        elif data.shape[2] == 3:
            data = data.transpose(2, 0, 1, 3)
        else:
            raise AssertionError(f"Expected 3 components on axis 0 or 2, got shape {data.shape}")

        T = data.shape[3]
        # Use exactly 5 fixed time indices
        sel_times = [0, T // 4, T // 2, (3 * T) // 4, T - 1]
        sel_times = [int(t) for t in sel_times if 0 <= t < T]
        assert len(sel_times) == 5, "Expected to plot exactly 5 time indices"

        # Use vector magnitude across components
        magnitude = np.sqrt(np.sum(data ** 2, axis=0))  # (H, W, T)
        # Robust global limits using percentiles
        vmin = np.percentile(magnitude, 1)
        vmax = np.percentile(magnitude, 99)

        num = len(sel_times)
        nrows, ncols = 1, 5
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        axes = np.array(axes).reshape(-1)

        for ax, t in zip(axes, sel_times):
            ax.imshow(magnitude[:, :, t], cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='equal', origin='lower')
            ax.set_title(f't = {t} ( {t * 0.25:.2f} s )')
            ax.set_xlabel('Width (stations)')
            ax.set_ylabel('Height (stations)')
        plt.tight_layout()
        # No unused axes for 1x5 fixed layout

        # No colorbar

        # Compose hypocenter text
        hypo_text = None
        try:
            if isinstance(hypo, np.ndarray) and len(hypo) > idx:
                val = hypo[idx]
                if isinstance(val, (list, tuple, np.ndarray)):
                    val_arr = np.array(val).astype(float).ravel()
                    hypo_text = np.array2string(val_arr, precision=3, separator=', ')
                else:
                    hypo_text = str(val)
        except Exception:
            hypo_text = None

        title = f'Slices of {npy_path.name}'
        if hypo_text is not None:
            title += f"\nHypocenter: {hypo_text}"
        fig.suptitle(title)
        # Leave space for the suptitle
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.92])
        out_path = save_dir / f"{npy_path.stem}_slices.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved slices figure to {out_path}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "main"
    if mode == "main":
        # Optional CLI: python rupture_process_M7_all.py main [start_idx] [max_count]
        s = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        m = int(sys.argv[3]) if len(sys.argv) > 3 else None
        process_rupture_data(start_idx=s, max_count=m)
    elif mode == "test":
        test_processed_sample()
    else:
        print("Usage: python rupture_process_M7_all.py [main|test]")
        sys.exit(1)


