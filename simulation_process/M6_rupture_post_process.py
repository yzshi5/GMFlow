## This file is used for two purporse : 
# 1. truncate the [x, y, t] to smaller one [320, 160, 96] and [256, 128, 96]
# 2. process outputs at [320, 160, 96] and [256, 128, 96]


import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import sys

# -----------------------------------------------------------------------------
# PATH CONFIG (edit once here; optional env vars override these defaults)
# -----------------------------------------------------------------------------
# Example:
#   export GMFLOW_PROCESSED_ROOT="/path/to/your/processed_output"
PROCESSED_ROOT = Path(os.getenv('GMFLOW_PROCESSED_ROOT', 'path/to/your/processed_output'))
M6_PROCESSED_DIR = PROCESSED_ROOT / 'processed_M6'
M6_PROCESSED_RAW_DIR = M6_PROCESSED_DIR / 'raw'
M6_PROC_320_160_DIR = M6_PROCESSED_DIR / 'm6_320_160'
M6_PROC_256_128_DIR = M6_PROCESSED_DIR / 'm6_256_128'


def _ensure_paths_configured() -> None:
    if 'path/to/your' in str(PROCESSED_ROOT):
        raise ValueError(
            'Please set PROCESSED_ROOT in this file or via env var '
            '(GMFLOW_PROCESSED_ROOT) before running.'
        )


data_path_M6 = M6_PROCESSED_RAW_DIR

save_path_M6_320_160 = M6_PROC_320_160_DIR
save_path_M6_256_128 = M6_PROC_256_128_DIR

## 16, 96
npts = 96
start_idx = 16

def process_all():
    _ensure_paths_configured()
    os.makedirs(save_path_M6_320_160, exist_ok=True)
    os.makedirs(save_path_M6_256_128, exist_ok=True)
    # original data shape is [3, 367, 171, 118] 
    files = [
        f for f in os.listdir(data_path_M6)
        if f.endswith('.npy') and not f.startswith('._') and not f.startswith('.')
    ]
    files.sort()
    for file in tqdm(files):
        data = np.load(os.path.join(data_path_M6, file), allow_pickle=False)

        # after truncation, [3, 320, 160, 96]
        data = data[:, 23:-24, 6:-5, start_idx:start_idx+npts]
        data = data.astype(np.float32, copy=False)

        # interpolate to [256, 128, 96]
        # data: [C, X, Y, T] -> [T, C, X, Y] for interpolation over (X, Y)
        _data = torch.as_tensor(data, dtype=torch.float32).permute(3, 0, 1, 2)
        _data = F.interpolate(_data, size=(256, 128), mode='bicubic', antialias=True)
        # back to [C, 256, 128, T]
        data_256 = _data.permute(1, 2, 3, 0).cpu().numpy()

        # save the data with standardized names: m6_proc_r_*_sim_{idx}.npy (float32)
        stem = Path(file).stem  # e.g., m6_raw_r_sim_0
        sim_str = "unknown"
        if "_sim_" in stem:
            tail = stem.split("_sim_")[-1]
            # keep only leading digits for safety
            num = ""
            for ch in tail:
                if ch.isdigit():
                    num += ch
                else:
                    break
            if len(num) > 0:
                sim_str = num
            else:
                sim_str = tail  # fallback to raw tail

        out_320 = f"m6_proc_r_320_160_sim_{sim_str}.npy"
        out_256 = f"m6_proc_r_256_128_sim_{sim_str}.npy"
        np.save(save_path_M6_320_160 / out_320, data.astype(np.float32, copy=False))
        np.save(save_path_M6_256_128 / out_256, data_256.astype(np.float32, copy=False))


def test_single_sample(
    raw_dir: Path = data_path_M6,
    out_320_dir: Path = save_path_M6_320_160,
    out_256_dir: Path = save_path_M6_256_128,
    sim_idx: int = 0,
):
    """
    Process one sample:
    raw -> [320,160,96] -> [256,128,96].
    """
    _ensure_paths_configured()
    out_320_dir.mkdir(parents=True, exist_ok=True)
    out_256_dir.mkdir(parents=True, exist_ok=True)

    raw_name = f"m6_raw_r_sim_{sim_idx}.npy"
    raw_path = raw_dir / raw_name
    assert raw_path.exists(), f"Raw file not found: {raw_path}"

    data = np.load(raw_path, allow_pickle=False)  # [3, 367, 171, 118]
    # crop to [3, 320, 160, 96]
    _data_crop = data[:, 23:-24, 6:-5, start_idx:start_idx+npts].astype(np.float32, copy=False)
    # interpolate over (X,Y) to [256,128] per time frame (keep C,T)
    _torch = torch.as_tensor(_data_crop, dtype=torch.float32).permute(3, 0, 1, 2)  # [T,C,X,Y]
    _torch = F.interpolate(_torch, size=(256, 128), mode='bicubic', antialias=True)
    data_256 = _torch.permute(1, 2, 3, 0).cpu().numpy()  # [C,256,128,96]

    # save processed outputs
    out_name_320 = f"m6_proc_r_320_160_sim_{sim_idx}.npy"
    out_name_256 = f"m6_proc_r_256_128_sim_{sim_idx}.npy"
    np.save(out_320_dir / out_name_320, _data_crop.astype(np.float32, copy=False))
    np.save(out_256_dir / out_name_256, data_256.astype(np.float32, copy=False))
    print(f"Saved: {out_name_320}, {out_name_256}")
    print(f"Shapes -> 320: {_data_crop.shape}, 256: {data_256.shape}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "run":
        # Run full directory processing
        process_all()
    elif mode == "test":
        sim_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        test_single_sample(sim_idx=sim_idx)
    else:
        print("Usage:")
        print("  python rupture_post_process_M6.py run")
        print("  python rupture_post_process_M6.py test [sim_idx]")


