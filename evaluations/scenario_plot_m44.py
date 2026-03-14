#%%
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple, Dict, Any

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import FuncFormatter
from torch.utils.data import Dataset

#%%

device = "cuda:1" if torch.cuda.is_available() else "cpu"
#post_fix = "x_pred_unif_filter_05_t48_double" # 1c_add, 1c_add_tr
ground_truth_path = Path('./save_plot/M44_scen_test/e2e_sample_200_ground_truth.npy')
synthetic_path = Path('./save_plot/M44_scen_test/e2e_sample_200_synthetic_100.npy')



#ground_truth_path = Path('./save_plot/M7_scen_test/e2e_sample_145_ground_truth_unbias.npy')
#synthetic_path = Path('./save_plot/M7_scen_test/e2e_sample_145_synthetic_100_unbias.npy')


# groudn_truth : [1, 3, 256, 128, 96]
# synthetic : [100, 3, 256, 128, 96]
ground_truth = np.load(ground_truth_path)
synthetic = np.load(synthetic_path)

# 
ground_truth = np.flip(ground_truth, axis=-2)
synthetic = np.flip(synthetic, axis=-2)

#%%
# Select one synthetic event to visualize
# im = ax.imshow(plot_data[:, :, frame], cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='equal', origin='lower')v

event_id = 30
if event_id < 0 or event_id >= synthetic.shape[0]:
    raise ValueError(f"event_id {event_id} out of range for synthetic data")


def magnitude_from_wavefield(wavefield: np.ndarray) -> np.ndarray:
    """Compute magnitude from [3, H, W, T] wavefield -> [H, W, T]."""
    if wavefield.shape[0] != 3:
        raise ValueError(f"Expected 3 components, got {wavefield.shape[0]}")
    return np.sqrt(
        wavefield[0] ** 2 + wavefield[1] ** 2 + wavefield[2] ** 2
    )


def time_to_index(times: np.ndarray, dt: float, n_frames: int) -> np.ndarray:
    """Convert times (s) to 0-based frame indices, clamped to [0, n_frames-1]."""
    indices = np.floor(times / dt + 1e-8).astype(int) - 1
    return np.clip(indices, 0, n_frames - 1)


def _scientific_tick_formatter(vmin: float, vmax: float) -> FuncFormatter:
    """Create a tick formatter like 0.8e-3 with adaptive exponent."""
    ref_abs = max(abs(vmin), abs(vmax))
    if ref_abs > 0.0:
        exponent = int(np.floor(np.log10(ref_abs)))
    else:
        exponent = 0
    scale = 10.0 ** exponent
    if exponent == 0:
        return FuncFormatter(lambda val, _: f"{val:.1f}")
    return FuncFormatter(lambda val, _: f"{val / scale:.1f}e{exponent}")


def _style_scientific_colorbar(
    cbar, vmin: float, vmax: float, n_ticks: int = 4, labelsize: int = 12
) -> None:
    """Apply consistent scientific tick styling to a colorbar."""
    ticks = np.linspace(vmin, vmax, n_ticks)
    cbar.set_ticks(ticks)
    cbar.formatter = _scientific_tick_formatter(vmin, vmax)
    cbar.update_ticks()
    cbar.ax.yaxis.get_offset_text().set_visible(False)
    cbar.ax.tick_params(labelsize=labelsize)


def plot_snapshot_grid(
    data: np.ndarray,
    times: np.ndarray,
    indices: np.ndarray,
    title: str,
    save_path: Path,
    vmin: float,
    vmax: float,
) -> None:
    """Plot 3x3 snapshot grid for selected time indices."""
    nrows, ncols = 3, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 14), constrained_layout=False)
    extent = [0.0, 40.0, 0.0, 80.0]

    for i, (ax, t, idx) in enumerate(zip(axes.flat, times, indices)):
        im = ax.imshow(
            data[:, :, idx],
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            aspect="equal",
            extent=extent,
        )
        ax.set_title(f"Time: {t:.1f}s", fontsize=16)
        ax.set_xlabel("X (km)", fontsize=16)
        ax.set_ylabel("Y (km)", fontsize=16)
        ax.set_xlim(0.0, 40.0)
        ax.set_ylim(0.0, 80.0)
        row = i // ncols
        col = i % ncols
        if row == nrows - 1:
            ax.set_xticks(ticks=[10, 20, 30, 40])
            ax.tick_params(axis="x", labelsize=14)
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])
        if col == 0:
            ax.set_yticks(ticks=[20, 40, 60, 80])
            ax.tick_params(axis="y", labelsize=14)
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])
        if col == 0 and row != nrows - 1:
            ax.xaxis.set_visible(False)
        if row == nrows - 1 and col != 0:
            ax.yaxis.set_visible(False)
        if row != nrows - 1 and col != 0:
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

    for row in range(nrows):
        im = axes[row, -1].images[0]
        cax = inset_axes(
            axes[row, -1],
            width="8%",
            height="100%",
            loc="lower left",
            bbox_to_anchor=(1.02, 0.0, 1.0, 1.0),
            bbox_transform=axes[row, -1].transAxes,
            borderpad=0.0,
        )
        cbar = fig.colorbar(im, cax=cax)
        _style_scientific_colorbar(cbar, vmin, vmax, n_ticks=4, labelsize=13)
        # cbar.set_label("(m/s)")

    fig.suptitle(title, fontsize=24)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05, wspace=0.02, hspace=0.2)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")


# %%

# Snapshot comparison along time dimension
dt_snapshot = 0.25
#times_sec = np.arange(0.5, 22.5 + 1e-6, 2.0)
times_sec = np.arange(1, 23+1e-6, 2.0)

gt_mag = magnitude_from_wavefield(ground_truth[0])
syn_mag = magnitude_from_wavefield(synthetic[event_id])
time_indices = time_to_index(times_sec, dt_snapshot, gt_mag.shape[-1])

#%%
gt_sel = gt_mag[:, :, time_indices]
syn_sel = syn_mag[:, :, time_indices]
all_sel = np.concatenate([gt_sel.ravel(), syn_sel.ravel()])
vmin = np.percentile(all_sel, 1)
vmax = np.percentile(all_sel, 99)

plot_snapshot_grid(
    gt_mag,
    times_sec,
    time_indices,
    title="     Ground Truth", # Magnitude Snapshots",
    save_path=Path("./save_plot/M44_scen_test/snapshots_gt_magnitude.png"),
    vmin=vmin,
    vmax=vmax,
)
plot_snapshot_grid(
    syn_mag,
    times_sec,
    time_indices,
    title=f"    Synthetics",#Magnitude Snapshots (event {event_id})",
    save_path=Path(
        f"./save_plot/M44_scen_test/snapshots_synthetic_event{event_id}_magnitude.png"
    ),
    vmin=vmin,
    vmax=vmax,
)

#%% plot PGV distribution 

def pgv_from_wavefield(wavefield: np.ndarray, mode: str = "full") -> np.ndarray:
    """Compute PGV from [..., 3, H, W, T] wavefield -> [..., H, W]."""
    if wavefield.shape[-4] != 3:
        raise ValueError(f"Expected 3 components, got {wavefield.shape[-4]}")
    if mode == "horizontal":
        mag = np.sqrt(wavefield[..., 0, :, :, :] ** 2 + wavefield[..., 1, :, :, :] ** 2)
    elif mode == "full":
        mag = np.sqrt(
            wavefield[..., 0, :, :, :] ** 2
            + wavefield[..., 1, :, :, :] ** 2
            + wavefield[..., 2, :, :, :] ** 2
        )
    else:
        raise ValueError("mode must be 'horizontal' or 'full'")
    return np.max(mag, axis=-1)

pgv_mode = "full"  # "horizontal" or "full"
gt_pgv = pgv_from_wavefield(ground_truth[0], mode=pgv_mode)
syn_pgv_single = pgv_from_wavefield(synthetic[event_id], mode=pgv_mode)
syn_pgv_all = pgv_from_wavefield(synthetic, mode=pgv_mode)
syn_pgv_mean = np.mean(syn_pgv_all, axis=0)
syn_pgv_std = np.std(syn_pgv_all, axis=0)

pgv_vmin = min(np.min(gt_pgv), np.min(syn_pgv_single), np.min(syn_pgv_mean))
pgv_vmax = max(np.max(gt_pgv), np.max(syn_pgv_single), np.max(syn_pgv_mean))
std_vmin = np.min(syn_pgv_std)
std_vmax = np.max(syn_pgv_std)

# Use log-scale color normalization to enhance contrast
pgv_norm = Normalize(vmin=pgv_vmin, vmax=pgv_vmax)
std_norm = Normalize(vmin=std_vmin, vmax=std_vmax)  
# eps = 1e-8
# pgv_norm = LogNorm(vmin=max(pgv_vmin, eps), vmax=pgv_vmax)
# std_norm = LogNorm(vmin=max(std_vmin, eps), vmax=std_vmax)
#%%
def plot_pgv_panels(
    gt_pgv: np.ndarray,
    syn_pgv_single: np.ndarray,
    syn_pgv_mean: np.ndarray,
    syn_pgv_std: np.ndarray,
    pgv_norm: Normalize,
    std_norm: Normalize,
    save_path: Path,
) -> None:
    """Plot PGV panels and save to disk."""
    fig, axes = plt.subplots(1, 4, figsize=(10, 4), constrained_layout=True)
    fig.suptitle("Peak Ground Velocity (PGV) Comparison", fontsize=18)
    im0 = axes[0].imshow(
        gt_pgv,
        cmap="coolwarm",
        norm=pgv_norm,
        aspect="equal",
        origin="lower",
    )
    axes[0].set_title("Ground Truth", fontsize=16)
    #axes[0].set_ylabel("PGV (m/s)")
    axes[0].axis("off")

    im1 = axes[1].imshow(
        syn_pgv_single,
        cmap="coolwarm",
        norm=pgv_norm,
        aspect="equal",
        origin="lower",
    )
    axes[1].set_title("Synthetic", fontsize=16)
    axes[1].axis("off")

    im2 = axes[2].imshow(
        syn_pgv_mean,
        cmap="coolwarm",
        norm=pgv_norm,
        aspect="equal",
        origin="lower",
    )
    axes[2].set_title("Synthetic - Mean", fontsize=16)
    axes[2].axis("off")

    im3 = axes[3].imshow(
        syn_pgv_std,
        cmap="coolwarm",
        norm=std_norm,
        aspect="equal",
        origin="lower",
    )
    axes[3].set_title("Synthetic - Uncertainty", fontsize=16)
    axes[3].axis("off")

    cbar_pgv = fig.colorbar(im2, ax=axes[:3], fraction=0.12, pad=0.02)
    _style_scientific_colorbar(
        cbar_pgv, float(pgv_norm.vmin), float(pgv_norm.vmax), n_ticks=4, labelsize=12
    )
    cbar_std = fig.colorbar(im3, ax=axes[3], fraction=0.12, pad=0.02)
    _style_scientific_colorbar(
        cbar_std, float(std_norm.vmin), float(std_norm.vmax), n_ticks=4, labelsize=12
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    #plt.close(fig)


pgv_save_path = Path("./save_plot/M44_scen_test/pgv_gt_vs_syn_stats_horizontal.png")
plot_pgv_panels(
    gt_pgv=gt_pgv,
    syn_pgv_single=syn_pgv_single,
    syn_pgv_mean=syn_pgv_mean,
    syn_pgv_std=syn_pgv_std,
    pgv_norm=pgv_norm,
    std_norm=std_norm,
    save_path=pgv_save_path,
)

# %%
# Plot time-series at selected stations: GT vs synthetic mean/std
#plt.close(fig_ts)


#%%


def fas_from_velocity_wavefield(
    wavefield: np.ndarray,
    dt: float,
    mode: str = "horizontal",
    use_acceleration: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute FAS map from velocity wavefield [3, H, W, T]."""
    if wavefield.shape[0] != 3:
        raise ValueError(f"Expected 3 components, got {wavefield.shape[0]}")
    data = (
        np.gradient(wavefield, dt, axis=-1) if use_acceleration else wavefield
    )
    fft = np.fft.rfft(data, axis=-1)
    fas_comp = np.abs(fft) * dt
    freqs = np.fft.rfftfreq(data.shape[-1], d=dt)

    if mode == "horizontal":
        fas = np.sqrt(0.5 * (fas_comp[0] ** 2 + fas_comp[1] ** 2))
    elif mode == "full":
        fas = np.sqrt((fas_comp[0] ** 2 + fas_comp[1] ** 2 + fas_comp[2] ** 2) / 3.0)
    else:
        raise ValueError("mode must be 'horizontal' or 'full'")
    return freqs, fas

# def fas_from_velocity_wavefield(
#     wavefield: np.ndarray,
#     dt: float,
#     mode: str = "horizontal",
#     use_acceleration: bool = False,
# ) -> tuple[np.ndarray, np.ndarray]:
#     """Compute FAS map from velocity wavefield [3, H, W, T]."""
#     if wavefield.shape[0] != 3:
#         raise ValueError(f"Expected 3 components, got {wavefield.shape[0]}")
#     data = (
#         np.gradient(wavefield, dt, axis=-1) if use_acceleration else wavefield
#     )
#     if mode == "horizontal":
#         mag = np.sqrt(data[0] ** 2 + data[1] ** 2)
#     elif mode == "full":
#         mag = np.sqrt(data[0] ** 2 + data[1] ** 2 + data[2] ** 2)
#     else:
#         raise ValueError("mode must be 'horizontal' or 'full'")
#     fft = np.fft.rfft(mag, axis=-1)
#     fas = np.abs(fft) * dt
#     freqs = np.fft.rfftfreq(mag.shape[-1], d=dt)
#     return freqs, fas


def fas_map_at_freq(
    wavefield: np.ndarray, dt: float, freq_target: float, mode: str = "horizontal"
) -> tuple[float, np.ndarray]:
    """Return FAS map at nearest frequency to freq_target."""
    freqs, fas = fas_from_velocity_wavefield(wavefield, dt=dt, mode=mode)
    idx = int(np.argmin(np.abs(freqs - freq_target)))
    return float(freqs[idx]), fas[:, :, idx]


def plot_fas_panels(
    gt_map: np.ndarray,
    syn_single_map: np.ndarray,
    syn_power_mean: np.ndarray,
    syn_std: np.ndarray,
    norm: Normalize,
    std_norm: Normalize,
    save_path: Path,
    freq_label: float,
) -> None:
    """Plot FAS panels and save to disk."""
    fig, axes = plt.subplots(1, 4, figsize=(10, 4), constrained_layout=True)
    fig.suptitle(
        f"Fourier Amplitude Spectrum (FAS) Comparison ({freq_label:.2f} Hz)",
        fontsize=18,
    )

    im0 = axes[0].imshow(gt_map, cmap="coolwarm", norm=norm, aspect="equal", origin="lower")
    axes[0].set_title("Ground Truth", fontsize=16)
    axes[0].axis("off")

    im1 = axes[1].imshow(
        syn_single_map, cmap="coolwarm", norm=norm, aspect="equal", origin="lower"
    )
    axes[1].set_title("Synthetic", fontsize=16)
    axes[1].axis("off")

    im2 = axes[2].imshow(
        syn_power_mean, cmap="coolwarm", norm=norm, aspect="equal", origin="lower"
    )
    axes[2].set_title("Synthetic - Mean", fontsize=16)
    axes[2].axis("off")

    im3 = axes[3].imshow(
        syn_std, cmap="coolwarm", norm=std_norm, aspect="equal", origin="lower"
    )
    axes[3].set_title("Synthetic - Uncertainty", fontsize=16)
    axes[3].axis("off")

    cbar_fas = fig.colorbar(im2, ax=axes[:3], fraction=0.12, pad=0.02)
    _style_scientific_colorbar(
        cbar_fas, float(norm.vmin), float(norm.vmax), n_ticks=4, labelsize=12
    )
    cbar_fas_std = fig.colorbar(im3, ax=axes[3], fraction=0.12, pad=0.02)
    _style_scientific_colorbar(
        cbar_fas_std, float(std_norm.vmin), float(std_norm.vmax), n_ticks=4, labelsize=12
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    #plt.close(fig)



# %%
# Fourier amplitude spectrum (FAS)

# FAS horizontal component maps at specific frequencies (power-mean over synthetics)
dt_fas = 0.25
fas_freqs = [0.25, 0.75]

for freq_target in fas_freqs:
    freq_gt, gt_fas_map = fas_map_at_freq(
        ground_truth[0], dt=dt_fas, freq_target=freq_target, mode="horizontal"
    )
    _, syn_single_map = fas_map_at_freq(
        synthetic[event_id], dt=dt_fas, freq_target=freq_target, mode="horizontal"
    )

    syn_maps = []
    for i in range(synthetic.shape[0]):
        _, syn_i_map = fas_map_at_freq(
            synthetic[i], dt=dt_fas, freq_target=freq_target, mode="horizontal"
        )
        syn_maps.append(syn_i_map)
    syn_maps = np.stack(syn_maps, axis=0)

    syn_power_mean = np.mean(syn_maps, axis=0)
    syn_std = np.std(syn_maps, axis=0)

    fas_vmin = min(np.min(gt_fas_map), np.min(syn_single_map), np.min(syn_power_mean))
    fas_vmax = max(np.max(gt_fas_map), np.max(syn_single_map), np.max(syn_power_mean))
    fas_norm = Normalize(vmin=fas_vmin, vmax=fas_vmax)
    fas_std_vmin = np.min(syn_std)
    fas_std_vmax = np.max(syn_std)
    fas_std_norm = Normalize(vmin=fas_std_vmin, vmax=fas_std_vmax)

    fas_save_path = Path(
        f"./save_plot/M44_scen_test/fas_horizontal_{freq_gt:.2f}hz_gt_vs_syn_stats.png"
    )
    plot_fas_panels(
        gt_map=gt_fas_map,
        syn_single_map=syn_single_map,
        syn_power_mean=syn_power_mean,
        syn_std=syn_std,
        norm=fas_norm,
        std_norm=fas_std_norm,
        save_path=fas_save_path,
        freq_label=freq_gt,
    )
# %%
# %% cross correlation


def ncc_ref_to_all_time(field, x0=0, y0=0, dt=0.25, max_lag_s=6.0, eps=1e-12, demean=False):
    """
    Time-domain 3C NCC between a reference station waveform and all stations in 'field'.

    Inputs:
      field: np.ndarray, shape [3, 256, 128, 96]
      x0, y0: reference station indices (e.g., corner 0,0)
      dt, max_lag_s, eps, demean: same meaning as above

    Returns:
      rho_max:  [256, 128] peak NCC over lags
      tau_star: [256, 128] best lag in seconds
    """
    C, X, Y, T = field.shape
    assert C == 3

    # Reference waveform: [3,T]
    ref = field[:, x0, y0, :].astype(np.float64, copy=False)

    # Flatten stations: [N,3,T]
    N = X * Y
    u = field.transpose(1, 2, 0, 3).reshape(N, C, T).astype(np.float64, copy=False)

    if demean:
        ref = ref - ref.mean(axis=-1, keepdims=True)
        u = u - u.mean(axis=-1, keepdims=True)

    K = int(round(max_lag_s / dt))
    K = min(K, T - 1)
    lags = np.arange(-K, K + 1, dtype=int)
    L = lags.size

    # Precompute energies
    ref2 = np.sum(ref * ref, axis=0)          # [T]
    u2 = np.sum(u * u, axis=1)                # [N,T]

    rho = np.empty((N, L), dtype=np.float64)

    for j, k in enumerate(lags):
        if k >= 0:
            rr = ref[:, :T - k]               # [3,To]
            uu = u[:, :, k:]                  # [N,3,To]
            Er = np.sum(ref2[:T - k])         # scalar
            Es = np.sum(u2[:, k:], axis=1)    # [N]
        else:
            kk = -k
            rr = ref[:, kk:]                  # [3,To]
            uu = u[:, :, :T - kk]             # [N,3,To]
            Er = np.sum(ref2[kk:])            # scalar
            Es = np.sum(u2[:, :T - kk], axis=1)  # [N]

        num = np.sum(uu * rr[None, :, :], axis=(1, 2))  # [N]
        den = np.sqrt(Er * Es) #+ eps
        rho[:, j] = num / den


    argmax = np.argmax(rho, axis=1)
    rho_max = rho[np.arange(N), argmax]
    k_star = lags[argmax]
    tau_star = k_star.astype(np.float64) * dt

    return rho_max.reshape(X, Y), tau_star.reshape(X, Y)

rho_map, tau_map = ncc_ref_to_all_time(
    ground_truth[0], x0=128, y0=64, dt=0.25, max_lag_s=6.0
)
rho_syn_map, tau_syn_map = ncc_ref_to_all_time(
    synthetic[event_id], x0=128, y0=64, dt=0.25, max_lag_s=6.0
)

rho_syn_all = []
tau_syn_all = []
for i in range(synthetic.shape[0]):
    rho_i, tau_i = ncc_ref_to_all_time(
        synthetic[i], x0=128, y0=64, dt=0.25, max_lag_s=6.0
    )
    rho_syn_all.append(rho_i)
    tau_syn_all.append(tau_i)
rho_syn_all = np.stack(rho_syn_all, axis=0)
tau_syn_all = np.stack(tau_syn_all, axis=0)
rho_syn_mean = np.mean(rho_syn_all, axis=0)
rho_syn_std = np.std(rho_syn_all, axis=0)
tau_syn_mean = np.mean(tau_syn_all, axis=0)
tau_syn_std = np.std(tau_syn_all, axis=0)

#%%


"""
   fig, axes = plt.subplots(1, 4, figsize=(10, 4), constrained_layout=True)
    fig.suptitle(
        f"Fourier Amplitude Spectrum (FAS) Comparison ({freq_label:.2f} Hz)",
        fontsize=18,
    )

    im0 = axes[0].imshow(gt_map, cmap="coolwarm", norm=norm, aspect="equal", origin="lower")
    axes[0].set_title("Ground Truth", fontsize=16)
    axes[0].axis("off")

    im1 = axes[1].imshow(
        syn_single_map, cmap="coolwarm", norm=norm, aspect="equal", origin="lower"
    )
    axes[1].set_title("Synthetic", fontsize=16)
    axes[1].axis("off")

    im2 = axes[2].imshow(
        syn_power_mean, cmap="coolwarm", norm=norm, aspect="equal", origin="lower"
    )
    axes[2].set_title("Synthetic - Mean", fontsize=16)
    axes[2].axis("off")

    im3 = axes[3].imshow(
        syn_std, cmap="coolwarm", norm=std_norm, aspect="equal", origin="lower"
    )
    axes[3].set_title("Synthetic - Uncertainty", fontsize=16)
    axes[3].axis("off")

    fig.colorbar(im2, ax=axes[:3], fraction=0.12, pad=0.02)
    cbar_std = fig.colorbar(im3, ax=axes[3], fraction=0.12, pad=0.02)
    _style_scientific_colorbar(
        cbar_std, float(std_vmin), float(std_vmax), n_ticks=4, labelsize=12
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    #plt.close(fig)

"""
def plot_ncc_panels(
    gt_map: np.ndarray,
    syn_map: np.ndarray,
    syn_mean: np.ndarray,
    syn_std: np.ndarray,
    vmin: float,
    vmax: float,
    std_vmin: float,
    std_vmax: float,
    titles: Tuple[str, str, str, str],
    suptitle: str,
    save_path: Path,
) -> None:
    """Plot NCC panels and save to disk."""
    fig, axes = plt.subplots(1, 4, figsize=(9.5, 4), constrained_layout=True)
    fig.suptitle(suptitle, fontsize=18)
    im0 = axes[0].imshow(
        gt_map, cmap="coolwarm", vmin=vmin, vmax=vmax, aspect="equal", origin="lower"
    )
    axes[0].set_title(titles[0], fontsize=16)
    axes[0].axis("off")

    im1 = axes[1].imshow(
        syn_map, cmap="coolwarm", vmin=vmin, vmax=vmax, aspect="equal", origin="lower"
    )
    axes[1].set_title(titles[1], fontsize=16)
    axes[1].axis("off")

    im2 = axes[2].imshow(
        syn_mean, cmap="coolwarm", vmin=vmin, vmax=vmax, aspect="equal", origin="lower"
    )
    axes[2].set_title(titles[2], fontsize=16)
    axes[2].axis("off")

    im3 = axes[3].imshow(
        syn_std,
        cmap="coolwarm",
        vmin=std_vmin,
        vmax=std_vmax,
        aspect="equal",
        origin="lower",
    )
    axes[3].set_title(titles[3], fontsize=16)
    axes[3].axis("off")

    cbar_main = fig.colorbar(im2, ax=axes[:3], fraction=0.12, pad=0.02)
    _style_scientific_colorbar(cbar_main, float(vmin), float(vmax), n_ticks=4, labelsize=12)
    cbar_std = fig.colorbar(im3, ax=axes[3], fraction=0.12, pad=0.02)
    _style_scientific_colorbar(
        cbar_std, float(std_vmin), float(std_vmax), n_ticks=4, labelsize=12
    )

    #plt.close(fig)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
# Plot NCC peak correlation maps
rho_vmin = min(np.min(rho_map), np.min(rho_syn_map), np.min(rho_syn_mean))
rho_vmax = max(np.max(rho_map), np.max(rho_syn_map), np.max(rho_syn_mean))
rho_std_vmin = np.min(rho_syn_std)
rho_std_vmax = np.max(rho_syn_std)
ncc_rho_save_path = Path("./save_plot/M44_scen_test/ncc_rho_gt_vs_syn_stats.png")

# Plot NCC lag maps
tau_vmin = min(np.min(tau_map), np.min(tau_syn_map), np.min(tau_syn_mean))
tau_vmax = max(np.max(tau_map), np.max(tau_syn_map), np.max(tau_syn_mean))
tau_std_vmin = np.min(tau_syn_std)
tau_std_vmax = np.max(tau_syn_std)
ncc_tau_save_path = Path("./save_plot/M44_scen_test/ncc_tau_gt_vs_syn_stats.png")


#%%
plot_ncc_panels(
    gt_map=rho_map,
    syn_map=rho_syn_map,
    syn_mean=rho_syn_mean,
    syn_std=rho_syn_std,
    vmin=rho_vmin,
    vmax=rho_vmax,
    std_vmin=rho_std_vmin,
    std_vmax=rho_std_vmax,
    titles=(
        "Ground Truth",
        "Synthetic",
        "Synthetic - Mean",
        "Synthetic - Uncertainty",
    ),
    suptitle="Peak Normalized Cross Correlation Coefficient Comparison",
    save_path=ncc_rho_save_path,
)

plot_ncc_panels(
    gt_map=tau_map,
    syn_map=tau_syn_map,
    syn_mean=tau_syn_mean,
    syn_std=tau_syn_std,
    vmin=tau_vmin,
    vmax=tau_vmax,
    std_vmin=tau_std_vmin,
    std_vmax=tau_std_vmax,
    titles=(
        "Ground Truth",
        "Synthetic",
        "Synthetic - Mean",
        "Synthetic - Uncertainty",
    ),
    suptitle="Normalized Cross Correlation Time Lag (s) Comparison",
    save_path=ncc_tau_save_path,
)
#plt.close(fig)
#%%
# Time-distance plot, work on gt_mag and syn_mag of shape [256, 128, 96] (x, y ,t) of (24 s)
def _format_time_distance_axis(ax: plt.Axes, distance_max: float) -> None:

    if distance_max == 40.0:
        ax.set_xlabel("Time (s)", fontsize=14)
        ax.set_ylabel("A1-A2 Distance (km)", fontsize=14)
    else:
        ax.set_xlabel("Time (s)", fontsize=14)
        ax.set_ylabel("B1-B2 Distance (km)", fontsize=14)
    ax.set_xticks([0, 6, 12, 18, 24])
    if distance_max == 40.0:
        ax.set_yticks([10, 20, 30, 40])
    else:
        ax.set_yticks([20, 40, 60, 80])
    ax.tick_params(axis="both", labelsize=12)


def plot_time_distance_slice_x(
    gt_mag: np.ndarray,
    syn_mag: np.ndarray,
    save_dir: Path,
    dt: float = 0.25,
) -> None:
    """Plot slice_x ([128,:,:]) as two separate figures (GT and Syn)."""
    gt_slice_x = gt_mag[128, :, :]   # [128, 96]
    syn_slice_x = syn_mag[128, :, :]
    T = gt_slice_x.shape[1]
    t0, t1 = 0.0, dt * T
    pair_vals_x = np.concatenate([gt_slice_x.ravel(), syn_slice_x.ravel()])
    vmin = np.percentile(pair_vals_x, 1)
    vmax = np.percentile(pair_vals_x, 99)

    save_dir.mkdir(parents=True, exist_ok=True)
    for data, title, filename in [
        (gt_slice_x, "Time-Distance plot - Ground Truth", "time_distance_slice_x_gt.png"),
        (syn_slice_x, "Time-Distance plot - Synthetic", "time_distance_slice_x_syn.png"),
    ]:
        fig, ax = plt.subplots(1, 1, figsize=(6, 2.5), constrained_layout=True)
        im = ax.imshow(
            data,
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            origin="lower",
            extent=[t0, t1, 0.0, 40.0],
        )
        ax.set_title(title, fontsize=16)
        _format_time_distance_axis(ax, distance_max=40.0)
        ax.set_xlim(t0, t1)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        _style_scientific_colorbar(cbar, float(vmin), float(vmax), n_ticks=4, labelsize=11)
        fig.savefig(save_dir / filename, dpi=150, bbox_inches="tight")


def plot_time_distance_slice_y(
    gt_mag: np.ndarray,
    syn_mag: np.ndarray,
    save_dir: Path,
    dt: float = 0.25,
) -> None:
    """Plot slice_y ([:,64,:]) as two separate figures (GT and Syn)."""
    gt_slice_y = gt_mag[:, 64, :]  # [256, 96]
    syn_slice_y = syn_mag[:, 64, :]
    T = gt_slice_y.shape[1]
    t0, t1 = 0.0, dt * T
    pair_vals_y = np.concatenate([gt_slice_y.ravel(), syn_slice_y.ravel()])
    vmin = np.percentile(pair_vals_y, 1)
    vmax = np.percentile(pair_vals_y, 99)

    for data, title, filename in [
        (gt_slice_y, "Time-Distance plot - Ground Truth", "time_distance_y_gt.png"),
        (syn_slice_y, "Time-Distance plot - Synthetic", "time_distance_y_syn.png"),
    ]:
        fig, ax = plt.subplots(1, 1, figsize=(6, 3.5), constrained_layout=True)
        im = ax.imshow(
            data,
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            origin="lower",
            extent=[t0, t1, 0.0, 80.0],
        )
        ax.set_title(title, fontsize=16)
        _format_time_distance_axis(ax, distance_max=80.0)
        ax.set_xlim(t0, t1)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        _style_scientific_colorbar(cbar, float(vmin), float(vmax), n_ticks=4, labelsize=11)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / filename, dpi=150, bbox_inches="tight")


save_dir = Path("./save_plot/M44_scen_test")
plot_time_distance_slice_x(gt_mag, syn_mag, save_dir, dt=0.25)
plot_time_distance_slice_y(gt_mag, syn_mag, save_dir, dt=0.25)


# %%
def select_random_stations(
    nx: int, ny: int, n_stations: int = 5, seed: int = 0
) -> List[Tuple[int, int, str]]:
    """Select random stations across the domain."""
    if n_stations <= 0:
        raise ValueError("n_stations must be positive")
    rng = np.random.default_rng(seed)
    total = nx * ny
    indices = rng.choice(total, size=n_stations, replace=False)
    xs = indices // ny
    ys = indices % ny
    stations: List[Tuple[int, int, str]] = []
    for i, (x, y) in enumerate(zip(xs, ys), start=1):
        stations.append((int(x), int(y), f"sta{i}"))
    return stations


def plot_station_mask(
    mask: np.ndarray,
    stations: Sequence[Tuple[int, int, str]],
    save_path: Path,
    highlight_label: str,
) -> None:
    """Plot station mask with one selected station highlighted in red."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ny, nx = mask.shape[1], mask.shape[0]
    x_max_km, y_max_km = 40.0, 80.0
    ax.imshow(
        mask,
        cmap="gray",
        origin="lower",
        aspect="equal",
        alpha=0.0,
        extent=[0.0, x_max_km, 0.0, y_max_km],
    )

    for x, y, label in stations:
        x_km = (y / (ny - 1)) * x_max_km
        y_km = (x / (nx - 1)) * y_max_km
        if label == highlight_label:
            ax.scatter(
                x_km,
                y_km,
                c="red",
                s=80,
                marker="^",
                edgecolors="black",
                linewidths=0.5,
                zorder=3,
            )
            ax.text(x_km + 0.5, y_km + 0.5, label, color="red", fontsize=11)
        else:
            ax.scatter(
                x_km,
                y_km,
                c="gray",
                s=45,
                marker="^",
                edgecolors="black",
                linewidths=0.3,
                alpha=0.8,
            )

    ax.set_xticks([10, 20, 30, 40])
    ax.set_yticks([20, 40, 60, 80])
    ax.set_xlabel("X (km)", fontsize=14)
    ax.set_ylabel("Y (km)", fontsize=14)
    ax.set_title("Mw 4.4")
    plt.tight_layout(pad=0.0)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", transparent=True)


# %%
# Additional random-reference analysis (keeps original center-reference analysis unchanged)
random_save_dir = Path("./save_plot/M44_scen_test")
nx, ny = ground_truth.shape[2], ground_truth.shape[3]
random_stations = select_random_stations(nx, ny, n_stations=4, seed=200)
station_mask = np.ones((nx, ny), dtype=np.float32)

for x0, y0, sta_label in random_stations:
    rho_rand_map, tau_rand_map = ncc_ref_to_all_time(
        ground_truth[0], x0=x0, y0=y0, dt=0.25, max_lag_s=6.0
    )
    rho_rand_syn_map, tau_rand_syn_map = ncc_ref_to_all_time(
        synthetic[event_id], x0=x0, y0=y0, dt=0.25, max_lag_s=6.0
    )

    rho_rand_syn_all = []
    tau_rand_syn_all = []
    for i in range(synthetic.shape[0]):
        rho_i, tau_i = ncc_ref_to_all_time(
            synthetic[i], x0=x0, y0=y0, dt=0.25, max_lag_s=6.0
        )
        rho_rand_syn_all.append(rho_i)
        tau_rand_syn_all.append(tau_i)

    rho_rand_syn_all = np.stack(rho_rand_syn_all, axis=0)
    tau_rand_syn_all = np.stack(tau_rand_syn_all, axis=0)
    rho_rand_syn_mean = np.mean(rho_rand_syn_all, axis=0)
    rho_rand_syn_std = np.std(rho_rand_syn_all, axis=0)
    tau_rand_syn_mean = np.mean(tau_rand_syn_all, axis=0)
    tau_rand_syn_std = np.std(tau_rand_syn_all, axis=0)

    rho_vmin = min(np.min(rho_rand_map), np.min(rho_rand_syn_map), np.min(rho_rand_syn_mean))
    rho_vmax = max(np.max(rho_rand_map), np.max(rho_rand_syn_map), np.max(rho_rand_syn_mean))
    rho_std_vmin = np.min(rho_rand_syn_std)
    rho_std_vmax = np.max(rho_rand_syn_std)

    tau_vmin = min(np.min(tau_rand_map), np.min(tau_rand_syn_map), np.min(tau_rand_syn_mean))
    tau_vmax = max(np.max(tau_rand_map), np.max(tau_rand_syn_map), np.max(tau_rand_syn_mean))
    tau_std_vmin = np.min(tau_rand_syn_std)
    tau_std_vmax = np.max(tau_rand_syn_std)

    plot_ncc_panels(
        gt_map=rho_rand_map,
        syn_map=rho_rand_syn_map,
        syn_mean=rho_rand_syn_mean,
        syn_std=rho_rand_syn_std,
        vmin=rho_vmin,
        vmax=rho_vmax,
        std_vmin=rho_std_vmin,
        std_vmax=rho_std_vmax,
        titles=(
            "Ground Truth",
            "Synthetic",
            "Synthetic - Mean",
            "Synthetic - Uncertainty",
        ),
        suptitle=f"Peak Normalized Cross Correlation Coefficient Comparison ({sta_label})",
        save_path=random_save_dir / f"ncc_rho_gt_vs_syn_stats_{sta_label}.png",
    )

    plot_ncc_panels(
        gt_map=tau_rand_map,
        syn_map=tau_rand_syn_map,
        syn_mean=tau_rand_syn_mean,
        syn_std=tau_rand_syn_std,
        vmin=tau_vmin,
        vmax=tau_vmax,
        std_vmin=tau_std_vmin,
        std_vmax=tau_std_vmax,
        titles=(
            "Ground Truth",
            "Synthetic",
            "Synthetic - Mean",
            "Synthetic - Uncertainty",
        ),
        suptitle=f"Normalized Cross Correlation Time Lag (s) Comparison ({sta_label})",
        save_path=random_save_dir / f"ncc_tau_gt_vs_syn_stats_{sta_label}.png",
    )

    plot_station_mask(
        station_mask,
        random_stations,
        random_save_dir / f"station_mask_{sta_label}.png",
        highlight_label=sta_label,
    )

# %%
