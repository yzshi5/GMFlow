## Used for residual analysis of M7 # save the file in hdf5 format 
## Two metrics : FAS and RotD50
## 
"""
Created on Tue Sep  6 06:07:08 2022

@author: glavrent
"""
#%%
#load libraries
import os
import sys
import pathlib
import warnings
#arithmetic libraries
import numpy as np
import scipy
from scipy import interpolate as interp
from scipy.fft import fft, ifft, fftfreq
from scipy.stats import wasserstein_distance
#statistics libraries
import pandas as pd
#ploting libraries
import matplotlib.pyplot as plt
#data management libraries
import h5py
import torch
from typing import Tuple

#%%
def get_h5_tree(val, pre=''):
    """
    Recursively prints the hierarchy of an HDF5 group or file with indentation.
    """
    items = len(val)
    for key, item in val.items():
        items -= 1
        if items == 0:
            # Last item in the current group
            print(f"{pre}└── {key} ({item.__class__.__name__})")
            if isinstance(item, h5py.Group):
                get_h5_tree(item, pre + '    ')
        else:
            print(f"{pre}├── {key} ({item.__class__.__name__})")
            if isinstance(item, h5py.Group):
                get_h5_tree(item, pre + '│   ')

#f = h5py.File('./save_plot/scaling_subset.h5', 'r')
#f = h5py.File('./save_plot/scaling_subset_05_horizontal_new.h5', 'r')
f = h5py.File('./save_plot/scaling_subset_075_horizontal_new_unbias.h5', 'r')


def get_h5_tree(val, pre=''):
    """
    Recursively prints the hierarchy of an HDF5 group or file with indentation.
    """
    items = len(val)
    for key, item in val.items():
        items -= 1
        if items == 0:
            # Last item in the current group
            print(f"{pre}└── {key} ({item.__class__.__name__})")
            if isinstance(item, h5py.Group):
                get_h5_tree(item, pre + '    ')
        else:
            print(f"{pre}├── {key} ({item.__class__.__name__})")
            if isinstance(item, h5py.Group):
                get_h5_tree(item, pre + '│   ')




# %%
# for magnitude scaling plot
"""
├── conditions (Dataset) (500,4 )
├── freq_bins (Dataset) (4,) # 0.25, 0.5, 0.75, 1.0Hz
├── freq_targets (Dataset) (4,)
├── ground_truth (Group)
│   ├── fas (Dataset) (500,4,256,128)
│   └── pgv (Dataset) (500,256,128)
├── indices (Dataset) (500,) (event idx)
├── synthetic (Group)
│   ├── fas (Dataset) (500,4,256,128)
│   └── pgv (Dataset) (500,256,128)
└── synthetic_interpolated (Group)
    ├── conditions (Dataset) (1600, 4)
    ├── fas (Dataset) (1600, 4, 256, 128)
    ├── magnitudes (Dataset) (1600,) 
    ├── pgv (Dataset) (1600, 256, 128)
    ├── source_index (Dataset)(1600,)
    └── source_pool (Dataset)

"""

## print the dataset shape 
pgv_gt = np.array(f['ground_truth/pgv'])
pgv_syn = np.array(f['synthetic/pgv'])
pgv_syn_interp = np.array(f['synthetic_interpolated/pgv'])

fas_gt = np.array(f['ground_truth/fas'])
fas_syn = np.array(f['synthetic/fas'])
fas_syn_interp = np.array(f['synthetic_interpolated/fas'])

#%% plot the log10(PGV) (pointwise distribution)
gt_log_pgv_m44 = np.log10(pgv_gt[200:].reshape(-1))
syn_log_pgv_m44 = np.log10(pgv_syn[200:].reshape(-1))

gt_log_pgv_m6 = np.log10(pgv_gt[0:100].reshape(-1))
syn_log_pgv_m6 = np.log10(pgv_syn[0:100].reshape(-1))

gt_log_pgv_m7 = np.log10(pgv_gt[100:200].reshape(-1))
syn_log_pgv_m7 = np.log10(pgv_syn[100:200].reshape(-1))

gt_log_pgv_m44 = gt_log_pgv_m44[np.isfinite(gt_log_pgv_m44)]
syn_log_pgv_m44 = syn_log_pgv_m44[np.isfinite(syn_log_pgv_m44)]
gt_log_pgv_m6 = gt_log_pgv_m6[np.isfinite(gt_log_pgv_m6)]
syn_log_pgv_m6 = syn_log_pgv_m6[np.isfinite(syn_log_pgv_m6)]
gt_log_pgv_m7 = gt_log_pgv_m7[np.isfinite(gt_log_pgv_m7)]
syn_log_pgv_m7 = syn_log_pgv_m7[np.isfinite(syn_log_pgv_m7)]

fig, axes = plt.subplots(3, 1, figsize=(4,8), sharey=True, sharex=True)
fig.suptitle('      Pointwise log$_{10}$ (PGV) Distribution', fontsize=14)

axes[0].hist(gt_log_pgv_m44, bins=100, density=True, alpha=0.6, label='Data', color='tab:blue')
axes[0].hist(syn_log_pgv_m44, bins=100, density=True, alpha=0.6, label='Synthetic', color='tab:orange')
axes[0].set_title('Mw 4.4', fontsize=14)
#axes[0].set_xlabel('log$_{10}$ (PGV)', fontsize=12)
axes[0].set_ylabel('Density', fontsize=12)
axes[0].legend(fontsize=11, loc='upper left')

axes[1].hist(gt_log_pgv_m6, bins=100, density=True, alpha=0.6, label='Data', color='tab:blue')
axes[1].hist(syn_log_pgv_m6, bins=100, density=True, alpha=0.6, label='Synthetic', color='tab:orange')
axes[1].set_title('Mw 6', fontsize=12)
#axes[1].set_xlabel('log$_{10}$ (PGV)', fontsize=12)
axes[1].set_ylabel('Density', fontsize=12)
axes[1].legend(fontsize=11, loc='upper left')

axes[2].hist(gt_log_pgv_m7, bins=100, density=True, alpha=0.6, label='Data', color='tab:blue')
axes[2].hist(syn_log_pgv_m7, bins=100, density=True, alpha=0.6, label='Synthetic', color='tab:orange')
axes[2].set_title('Mw 7', fontsize=12)
axes[2].set_xlabel('log$_{10}$ (PGV)', fontsize=12)
axes[2].set_ylabel('Density', fontsize=12)
axes[2].legend(fontsize=11, loc='upper left')

fig.tight_layout()

#%% log10 (FAS) distribution for M4.4, M6, M7 of Frequency 0.25, 0.5, 0.75, 0.96 Hz (Fas_syn of shape [300, 5, 256, 128] (5 correpsonds to 0.25, 0.5, 0.75, 0.95, 1 Hz))
freq_indices = [0, 1, 2, 3]
freq_labels = ['0.25', '0.5', '0.75', '0.96']
mags = ['Mw 4.4', 'Mw 6', 'Mw 7']
mag_slices = [slice(200, None), slice(0, 100), slice(100, 200)]

fig, axes = plt.subplots(3, 4, figsize=(13, 8), sharex=True, sharey=True)
fig.suptitle('Pointwise log$_{10}$ (FAS) Distribution', fontsize=16)

for r, mag_slice in enumerate(mag_slices):
    for c, fi in enumerate(freq_indices):
        gt_vals = fas_gt[mag_slice, fi].reshape(-1)
        syn_vals = fas_syn[mag_slice, fi].reshape(-1)

        gt_vals = gt_vals[gt_vals > 0]
        syn_vals = syn_vals[syn_vals > 0]

        gt_log = np.log10(gt_vals)
        syn_log = np.log10(syn_vals)

        ax = axes[r, c]
        ax.hist(gt_log, bins=100, density=True, alpha=0.6, label='Data', color='tab:blue')
        ax.hist(syn_log, bins=100, density=True, alpha=0.6, label='Synthetic', color='C3')

        ax.set_title(f'{mags[r]} | {freq_labels[c]} Hz', fontsize=12)
        if c == 0:
            ax.set_ylabel('Density', fontsize=12)
        if r == 2:
            ax.set_xlabel('log$_{10}$ (FAS)', fontsize=12)
        if c == 0:
            ax.legend(fontsize=11, loc='upper left')

fig.tight_layout()

# plot the pgv and fas



#%%
def cal_gmean_gstd(wfs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate geometric mean and geometric std along dim=0.
    """
    log_wfs = torch.log(wfs)
    mean = torch.mean(log_wfs, dim=0)
    std = torch.std(log_wfs, dim=0)
    return torch.exp(mean).cpu(), torch.exp(std).cpu()


# --- Figure 1: PGV scaling vs magnitude ---
gt_mags = np.array(f['conditions'])[:, 3] * 10.0
gt_pgv_median = np.median(pgv_gt, axis=(1, 2))

syn_mags = np.array(f['conditions'])[:, 3] * 10.0
syn_pgv_median = np.median(pgv_syn, axis=(1, 2))

syn_interp_mags = np.array(f['synthetic_interpolated/magnitudes'])
syn_interp_pgv_median = np.median(pgv_syn_interp, axis=(1, 2))

all_syn_mags = np.concatenate([syn_mags, syn_interp_mags], axis=0)
all_syn_pgv = np.concatenate([syn_pgv_median, syn_interp_pgv_median], axis=0)

# expected target magnitudes with possible floating error
target_mags = np.arange(4.4, 7.1, 0.1, dtype=np.float32)
mag_tol = 1e-3

syn_mag_list = []
syn_gmean_list = []
syn_gstd_list = []

for mag in target_mags:
    mask = np.isclose(all_syn_mags, mag, atol=mag_tol)
    if not np.any(mask):
        continue
    wfs = torch.tensor(all_syn_pgv[mask], dtype=torch.float32)
    gmean, gstd = cal_gmean_gstd(wfs)
    syn_mag_list.append(mag)
    syn_gmean_list.append(gmean.item())
    syn_gstd_list.append(gstd.item())

syn_mag_arr = np.array(syn_mag_list)
syn_gmean_arr = np.array(syn_gmean_list)
syn_gstd_arr = np.array(syn_gstd_list)

# convert geometric std to asymmetric error bars
syn_lower = syn_gmean_arr / syn_gstd_arr
syn_upper = syn_gmean_arr * syn_gstd_arr
syn_yerr = np.vstack([syn_gmean_arr - syn_lower, syn_upper - syn_gmean_arr])

#%%
plt.figure(figsize=(6,5))
plt.semilogy(gt_mags, gt_pgv_median, 'o', ms=3, alpha=0.6, label='Data')
plt.errorbar(
    syn_mag_arr, syn_gmean_arr, yerr=syn_yerr,
    fmt='s', ms=5, capsize=3, label='Synthetic'
)
plt.xlabel('Magnitude', fontsize=14)
plt.ylabel('Amplitude (median over grid)', fontsize=10)
plt.title('PGV - Magnitude Scaling', fontsize=14)
plt.tick_params(axis='both', labelsize=10)
plt.grid(True, which='both', ls='--', alpha=0.3)
plt.legend(fontsize=11)
# %%
# %%

# --- FAS scaling vs magnitude at selected frequencies ---
freq_bins = np.array(f['freq_bins'])

subsampling_ratio = 1
for fi, freq in enumerate(freq_bins):
    gt_fas_median = np.median(fas_gt[:, fi, ::subsampling_ratio, ::subsampling_ratio], axis=(1, 2))
    syn_fas_median = np.median(fas_syn[:, fi, ::subsampling_ratio, ::subsampling_ratio], axis=(1, 2))
    syn_interp_fas_median = np.median(fas_syn_interp[:, fi, ::subsampling_ratio, ::subsampling_ratio], axis=(1, 2))

    all_syn_fas = np.concatenate([syn_fas_median, syn_interp_fas_median], axis=0)

    syn_mag_list = []
    syn_gmean_list = []
    syn_gstd_list = []

    for mag in target_mags:
        mask = np.isclose(all_syn_mags, mag, atol=mag_tol)
        if not np.any(mask):
            continue
        wfs = torch.tensor(all_syn_fas[mask], dtype=torch.float32)
        gmean, gstd = cal_gmean_gstd(wfs)
        syn_mag_list.append(mag)
        syn_gmean_list.append(gmean.item())
        syn_gstd_list.append(gstd.item())

    syn_mag_arr = np.array(syn_mag_list)
    syn_gmean_arr = np.array(syn_gmean_list)
    syn_gstd_arr = np.array(syn_gstd_list)

    syn_lower = syn_gmean_arr / syn_gstd_arr
    syn_upper = syn_gmean_arr * syn_gstd_arr
    syn_yerr = np.vstack([syn_gmean_arr - syn_lower, syn_upper - syn_gmean_arr])

    plt.figure(figsize=(4, 2.5))
    plt.semilogy(gt_mags, gt_fas_median, 'o', ms=3, alpha=0.6, label='Data')
    plt.errorbar(
        syn_mag_arr, syn_gmean_arr, yerr=syn_yerr,
        fmt='s', ms=5, capsize=3, label='Synthetic'
    )
    plt.xlabel('Magnitude')
    plt.ylabel('FAS')
    plt.title(f'FAS - Magnitude Scaling ({freq:.2f} Hz)')
    plt.grid(True, which='both', ls='--', alpha=0.3)
    plt.legend(fontsize=9)
    plt.tick_params(axis='both')


# %%
# %% Wasserstein distance in log10 domain (appended after scaling)
# Uses the same magnitude group slices and selected frequencies as above.
wd_mag_labels = ['Mw 4.4', 'Mw 6', 'Mw 7']
wd_mag_slices = [slice(200, None), slice(0, 100), slice(100, 200)]
wd_freq_indices = [0, 1, 2, 3]
wd_freq_labels = ['0.25', '0.5', '0.75', '0.96']

# --- Wasserstein distance: log10(PGV) ---
pgv_wdist = []
for mag_slice in wd_mag_slices:
    gt_vals = pgv_gt[mag_slice].reshape(-1)
    syn_vals = pgv_syn[mag_slice].reshape(-1)

    gt_vals = gt_vals[gt_vals > 0]
    syn_vals = syn_vals[syn_vals > 0]

    gt_log = np.log10(gt_vals)
    syn_log = np.log10(syn_vals)
    gt_log = gt_log[np.isfinite(gt_log)]
    syn_log = syn_log[np.isfinite(syn_log)]
    pgv_wdist.append(wasserstein_distance(gt_log, syn_log))

plt.figure(figsize=(5, 3.5))
plt.bar(wd_mag_labels, pgv_wdist, color='tab:purple', alpha=0.8)
plt.ylabel('Wasserstein distance', fontsize=12)
plt.title('log$_{10}$(PGV): Data vs Synthetic', fontsize=13)
plt.grid(axis='y', ls='--', alpha=0.3)
plt.tick_params(axis='both', labelsize=10)
plt.tight_layout()

# --- Wasserstein distance: log10(FAS) at selected frequencies ---
fas_wdist = np.zeros((len(wd_mag_slices), len(wd_freq_indices)))
for r, mag_slice in enumerate(wd_mag_slices):
    for c, fi in enumerate(wd_freq_indices):
        gt_vals = fas_gt[mag_slice, fi].reshape(-1)
        syn_vals = fas_syn[mag_slice, fi].reshape(-1)

        gt_vals = gt_vals[gt_vals > 0]
        syn_vals = syn_vals[syn_vals > 0]

        gt_log = np.log10(gt_vals)
        syn_log = np.log10(syn_vals)
        gt_log = gt_log[np.isfinite(gt_log)]
        syn_log = syn_log[np.isfinite(syn_log)]
        fas_wdist[r, c] = wasserstein_distance(gt_log, syn_log)

fig, ax = plt.subplots(figsize=(7, 3.8))
im = ax.imshow(fas_wdist, cmap='viridis', aspect='auto')
ax.set_xticks(np.arange(len(wd_freq_labels)))
ax.set_xticklabels(wd_freq_labels)
ax.set_yticks(np.arange(len(wd_mag_labels)))
ax.set_yticklabels(wd_mag_labels)
ax.set_xlabel('Frequency (Hz)', fontsize=12)
ax.set_title('log$_{10}$(FAS): Wasserstein distance', fontsize=13)

for i in range(fas_wdist.shape[0]):
    for j in range(fas_wdist.shape[1]):
        ax.text(j, i, f'{fas_wdist[i, j]:.3f}', ha='center', va='center', color='white', fontsize=9)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Wasserstein distance', fontsize=11)
fig.tight_layout()

# %%
print(pgv_wdist);
print(fas_wdist)
# %%
