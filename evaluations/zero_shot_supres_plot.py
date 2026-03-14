"""Snapshot-grid plotting for super-resolution outputs.

Expected arrays:
- data: [1, 3, 256, 128, 96]
- syn:  [1, 3, 512, 256, 96]
"""
#%%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes




# %%
##




OUTPUT_DIR = Path("./save_plot/supres")
EVENTS = [
    (75, "Mw 6.0"),
    (175, "Mw 7.0"),
    (275, "Mw 4.4"),
]


def magnitude_from_wavefield(wavefield: np.ndarray) -> np.ndarray:
    """Compute magnitude from [3, H, W, T] -> [H, W, T]."""
    if wavefield.shape[0] != 3:
        raise ValueError(f"Expected 3 components in axis 0, got {wavefield.shape[0]}")
    return np.sqrt(wavefield[0] ** 2 + wavefield[1] ** 2 + wavefield[2] ** 2)


def time_to_index(times: np.ndarray, dt: float, n_frames: int) -> np.ndarray:
    """Convert times in seconds to 0-based indices, clamped to frame range."""
    indices = np.floor(times / dt + 1e-8).astype(int) - 1
    return np.clip(indices, 0, n_frames - 1)


def _scientific_tick_formatter(vmin: float, vmax: float) -> FuncFormatter:
    ref_abs = max(abs(vmin), abs(vmax))
    exponent = int(np.floor(np.log10(ref_abs))) if ref_abs > 0.0 else 0
    scale = 10.0 ** exponent
    if exponent == 0:
        return FuncFormatter(lambda val, _: f"{val:.1f}")
    return FuncFormatter(lambda val, _: f"{val / scale:.1f}e{exponent}")


def _style_scientific_colorbar(cbar, vmin: float, vmax: float) -> None:
    ticks = np.linspace(vmin, vmax, 4)
    cbar.set_ticks(ticks)
    cbar.formatter = _scientific_tick_formatter(vmin, vmax)
    cbar.update_ticks()
    cbar.ax.yaxis.get_offset_text().set_visible(False)
    cbar.ax.tick_params(labelsize=11)


def plot_snapshot_grid(
    mag: np.ndarray,
    times: np.ndarray,
    indices: np.ndarray,
    title: str,
    save_path: Path,
    vmin: float,
    vmax: float,
) -> None:
    """Plot 3x4 snapshot grid for selected times."""
    nrows, ncols = 3, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 14), constrained_layout=False)
    # Keep the physical domain fixed across GT/SYN grids.
    extent = [0.0, 40.0, 0.0, 80.0]

    for i, (ax, t, idx) in enumerate(zip(axes.flat, times, indices)):
        im = ax.imshow(
            mag[:, :, idx],
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            aspect="equal",
            extent=extent,
        )
        ax.set_title(f"Time: {t:.1f}s", fontsize=14)
        ax.set_xlim(0.0, 40.0)
        ax.set_ylim(0.0, 80.0)
        row = i // ncols
        col = i % ncols
        if row == nrows - 1:
            ax.set_xlabel("X (km)", fontsize=12)
            ax.set_xticks([10, 20, 30, 40])
            ax.tick_params(axis="x", labelsize=11)
        else:
            ax.set_xticks([])
            ax.set_xticklabels([])
        if col == 0:
            ax.set_ylabel("Y (km)", fontsize=12)
            ax.set_yticks([20, 40, 60, 80])
            ax.tick_params(axis="y", labelsize=11)
        else:
            ax.set_yticks([])
            ax.set_yticklabels([])

    for row in range(nrows):
        row_im = axes[row, -1].images[0]
        cax = inset_axes(
            axes[row, -1],
            width="8%",
            height="100%",
            loc="lower left",
            bbox_to_anchor=(1.02, 0.0, 1.0, 1.0),
            bbox_transform=axes[row, -1].transAxes,
            borderpad=0.0,
        )
        cbar = fig.colorbar(row_im, cax=cax)
        _style_scientific_colorbar(cbar, vmin, vmax)

    fig.suptitle(title, fontsize=20)
    fig.subplots_adjust(left=0.06, right=0.93, top=0.93, bottom=0.06, wspace=0.05, hspace=0.2)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    #plt.close(fig)


def plot_single_time_snapshot(
    mag: np.ndarray,
    panel_title: str,
    time_sec: float,
    dt_snapshot: float,
    save_path: Path,
    vmin: float,
    vmax: float,
) -> None:
    """Plot one snapshot at a specific time."""
    idx = int(time_to_index(np.array([time_sec]), dt_snapshot, mag.shape[-1])[0])
    extent = [0.0, 40.0, 0.0, 80.0]

    fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.8), constrained_layout=False)
    im = ax.imshow(
        mag[:, :, idx],
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        aspect="equal",
        extent=extent,
    )
    ax.set_title(f"{panel_title} at {time_sec:.0f} (s)", fontsize=14)
    ax.set_xlabel("X (km)", fontsize=12)
    ax.set_ylabel("Y (km)", fontsize=12)
    ax.set_xlim(0.0, 40.0)
    ax.set_ylim(0.0, 80.0)
    ax.set_xticks([10, 20, 30, 40])
    ax.set_yticks([20, 40, 60, 80])
    ax.tick_params(labelsize=11)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    _style_scientific_colorbar(cbar, vmin, vmax)

    fig.subplots_adjust(left=0.12, right=0.90, top=0.98, bottom=0.14)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _select_wavefield(arr: np.ndarray) -> np.ndarray:
    """Normalize input to [3, H, W, T]."""
    if arr.ndim == 5:
        return arr[0]
    if arr.ndim == 4:
        return arr
    raise ValueError(f"Expected 4D or 5D array, got shape {arr.shape}")


def _resolve_existing_path(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find file. Tried:\n" + "\n".join(str(p) for p in candidates)
    )


def _resolve_event_paths(event_id: int) -> tuple[Path, Path]:
    gt_candidates = [
        OUTPUT_DIR / f"ground_truth_sample_{event_id}.npy",
    ]
    syn_candidates = [
        OUTPUT_DIR / f"sno_query_scale_2_denorm_sample_{event_id}.npy",

    ]
    return _resolve_existing_path(gt_candidates), _resolve_existing_path(syn_candidates)


def process_event(event_id: int, mw_label: str) -> None:
    data_path, syn_path = _resolve_event_paths(event_id)
    data = np.load(data_path)
    syn = np.load(syn_path)

    data = np.flip(data, axis=-2)
    syn = np.flip(syn, axis=-2)

    data_mag = magnitude_from_wavefield(_select_wavefield(data))
    syn_mag = magnitude_from_wavefield(_select_wavefield(syn))

    dt_snapshot = 0.25
    times_sec = np.arange(1.0, 23.0 + 1e-6, 2.0)
    time_indices = time_to_index(times_sec, dt_snapshot, data_mag.shape[-1])

    data_sel = data_mag[:, :, time_indices]
    syn_sel = syn_mag[:, :, time_indices]
    data_vmin = np.percentile(data_sel, 1)
    data_vmax = np.percentile(data_sel, 99)
    syn_vmin = np.percentile(syn_sel, 1)
    syn_vmax = np.percentile(syn_sel, 99)

    plot_snapshot_grid(
        data_mag,
        times_sec,
        time_indices,
        title=f"Ground Truth {mw_label} Snapshots",
        save_path=OUTPUT_DIR / f"snapshots_ground_truth_sample_{event_id}.png",
        vmin=data_vmin,
        vmax=data_vmax,
    )
    plot_snapshot_grid(
        syn_mag,
        times_sec,
        time_indices,
        title=f"Super-Resolution {mw_label} Snapshots",
        save_path=OUTPUT_DIR / f"snapshots_syn_sample_{event_id}.png",
        vmin=syn_vmin,
        vmax=syn_vmax,
    )



if __name__ == "__main__":
    for event_id, mw_label in EVENTS:
        process_event(event_id, mw_label)


# %%
