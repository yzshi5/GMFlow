## Used for residual analysis of M6 # save the file in hdf5 format 
## Metric: FAS only (FS)
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
#statistics libraries
import pandas as pd
#ploting libraries
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#data management libraries
import h5py
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

#f = h5py.File('./save_plot/M7_scen/m7_residual_data.h5', 'r')
#H5_PATH = './save_plot/residual_data_vel_05.h5'
H5_PATH = './save_plot/residual_data_vel_075_unbias.h5'


"""
├── synthetic (Group)
│   └── FS (Group)
│       ├── H_FS (Dataset) # (300, 2, 32768, 49)
│       ├── V_FS (Dataset) # (300, 2, 32768, 49) 
│       └── freq (Dataset)
└── validation (Group)
    └── FS (Group)
        ├── H_FS (Dataset) # (300, 32768, 49)
        ├── V_FS (Dataset) # (300, 32768, 49)
        └── freq (Dataset) # 49 [0~2 Hz]
"""

## To do 

# %%
# Quick log-log comparison for a single event/station
event_idx = 20

with h5py.File(H5_PATH, 'r') as f_resid:
    fas_freq = np.array(f_resid['synthetic/FS/freq'])
    syn_h_fs = np.array(f_resid['synthetic/FS/H_FS'])
    syn_v_fs = np.array(f_resid['synthetic/FS/V_FS'])
    gt_h_fs = np.array(f_resid['validation/FS/H_FS'])
    gt_v_fs = np.array(f_resid['validation/FS/V_FS'])

    obs_h_fs = np.array(f_resid['validation/FS/H_FS'])
    obs_v_fs = np.array(f_resid['validation/FS/V_FS'])

#%%



def PlotNewResVsFreq(freq, res_mu, res_sig=None, grp=None, 
                  x_label='X', y_label='Residuals',
                  grp_label=None,
                  flag_xlog=False, flag_norm=False, fLow=None, fHigh=None, periods=False):

    # flatten [batch, station, freq] -> [batch*station, freq]
    res_mu = np.asarray(res_mu)
    if res_mu.ndim == 3:
        res_mu = res_mu.reshape(-1, res_mu.shape[-1])
        if res_sig is not None:
            res_sig = np.asarray(res_sig).reshape(-1, res_sig.shape[-1])
        if grp is not None:
            grp = np.asarray(grp).reshape(-1)
        if fLow is not None:
            fLow = np.asarray(fLow).reshape(-1)
        if fHigh is not None:
            fHigh = np.asarray(fHigh).reshape(-1)

    #residuals grouping
    if grp is None:
        grp_array = np.ones(res_mu.shape[0])
    else:
        grp_array = grp
    grp = np.unique(grp_array)
    
    #print("Grp:",grp)
    #create figure
    fig, ax = plt.subplots(figsize = (6,5))
    for j, g in enumerate(grp):
        #select group's residuals, i_g = [true, true, true, ...., ]
        i_g = grp_array == g
        #print("i_g value: ", i_g)
        #print('before res_mu shape: ', res_mu.shape)
        res_g = res_mu[i_g,:]/res_sig[i_g,:] if flag_norm else res_mu[i_g,:]
        cur_fLow = None if fLow is None else fLow[i_g]
        cur_fHigh = None if fHigh is None else fHigh[i_g]
        #print('after res_g shape: ', res_g.shape)
        #group label
        lb_g = '' if grp_label is None else '%s - '%grp_label[j]
        
        mean_res_g = np.full(fill_value=np.nan, shape=freq.shape)
        median_res_g = np.full(fill_value=np.nan, shape=freq.shape)
        percent_res_g = np.full(fill_value=np.nan, shape=(len(freq),2))

        for k, cur_freq in enumerate(freq):
            if cur_fLow is None or cur_fHigh is None:
                index = np.ones(res_g.shape[0], dtype=bool)
            elif periods == False:
                index = (cur_freq>=cur_fLow) & (cur_freq<=cur_fHigh)
            else: 
                # cur_freq is actually period
                index = (1./cur_freq)>=cur_fLow
            # take corresponding slice
            cur_res_g = res_g[index][:,k]

            cur_mean = np.mean(cur_res_g)
            cur_median = np.median(cur_res_g)
            try:
                cur_percent = np.percentile(cur_res_g, [16,84])
            except:
                cur_percent = [np.nan,np.nan]
            mean_res_g[k] = cur_mean
            median_res_g[k] = cur_median
            percent_res_g[k,:] = cur_percent
            
            
        hl1 = ax.plot(freq, mean_res_g, 's',  markersize=4, zorder=3*(0+1), 
                          label=r'%sMean'%lb_g)[0]
        hl2 = ax.plot(freq, median_res_g, 'o',  markersize=4,
                          label=r'%sMedian'%lb_g)[0]
        hl3 = ax.errorbar(freq, median_res_g, 
                         yerr=np.abs(percent_res_g -median_res_g[:,np.newaxis]).T,
                         capsize=3, fmt='none', ecolor=hl2.get_color(),
                         label=r'%s16th/84th Prec'%lb_g)[0]

    
    #plot reference line
    ax.plot(freq, np.full(freq.shape,0.), color='black', linewidth=2)
    #edit properties
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    #plot log-scale
    if flag_xlog: 
        ax.set_xscale('log')
        ax.set_xticks([0.1, 0.2, 0.5, 0.96])
        ax.set_xticklabels([r'$10^{-1}$', r'$2\times10^{-1}$',
                            r'$5\times10^{-1}$', r'$10^{0}$'])
        # hide minor tick labels on log axis
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
    #add legend
    ax.legend(loc='lower right', fontsize=12).set_zorder(100)
    ax.grid(which='both')
    ax.tick_params(axis='x', which='both', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)

    return fig, ax


def ComputeResRMSE(freq_array, res_mat, log_space=True, flag_all=False):
    
    #mean squared errors
    res_rmse = np.mean(res_mat**2, axis=0)
    
    #compute average rmse from all frequencies
    if flag_all:
        #transform to log space if log_space is true
        x_array = np.log(freq_array) if log_space else freq_array
        if np.isinf(x_array[0]): x_array[0] = 2*x_array[1] - x_array[2]
        #compute mean squared error of all frequencies
        res_rmse = np.trapz(x=x_array, y=res_rmse) / np.trapz(x=x_array, y=np.ones(x_array.shape))

    return np.sqrt(res_rmse)





#%% Residual analysis for M6 (batch * station)

#%%
syn_h_fs_mean = syn_h_fs[:, 0]
syn_h_fs_std = syn_h_fs[:, 1]
syn_v_fs_mean = syn_v_fs[:, 0]
syn_v_fs_std = syn_v_fs[:, 1]

# residuals in log space
res_h_logfs = np.log(obs_h_fs) - np.log(syn_h_fs_mean)
res_v_logfs = np.log(obs_v_fs) - np.log(syn_v_fs_mean)


#%%
# normalized residuals (std is in log space for geometric std)
res_h_logfs_norm = res_h_logfs #[:, ::100] #/ np.log(syn_h_fs_std)
#%%
res_h_logfs_norm  = res_h_logfs_norm.reshape(-1, 256, 128, 49)
res_h_logfs_norm = np.flip(res_h_logfs_norm, axis=-2)
res_h_logfs_norm = res_h_logfs_norm.reshape(-1, 32768,49)

#%%
# Plot residuals by magnitude bins as separate figures:
# M6 (0-99), M7 (100-199), M4.4 (200-299)
max_events = min(300, res_h_logfs_norm.shape[0])
res_h_logfs_norm = res_h_logfs_norm[:max_events]

# RMSE of mean residual (target=0) across 0-0.96 Hz for each Mw bin
freq_mask_rmse = (fas_freq >= 0.1) & (fas_freq <= 0.96)
rmse_mean_residual_by_mag = {}
rmse_mag_slices = [
    ('Mw 4.4', 200, 300),
    ('Mw 6', 0, 100),
    ('Mw 7', 100, 200),
]
for mag_label, start_idx, end_idx in rmse_mag_slices:
    if start_idx >= max_events:
        continue
    stop_idx = min(end_idx, max_events)
    res_h_mag = res_h_logfs_norm[start_idx:stop_idx]  # [event, station, freq]
    mean_residual_freq = np.mean(res_h_mag, axis=(0, 1))
    rmse_mean_residual = np.sqrt(np.mean(mean_residual_freq[freq_mask_rmse] ** 2))
    rmse_mean_residual_by_mag[mag_label] = rmse_mean_residual
    print(f'RMSE(mean residual, 0.1-0.96 Hz) - {mag_label}: {rmse_mean_residual:.6f}')



#%%
mag_bins = [
    ('Mw 6', 0, 100),
    ('Mw 7', 100, 200),
    ('Mw 4.4', 200, 300),
]

freq_cutoff_hz = 1.0
cutoff_idx = np.searchsorted(fas_freq, freq_cutoff_hz, side='right')

for mag_label, start_idx, end_idx in mag_bins:
    if start_idx >= max_events:
        continue
    stop_idx = min(end_idx, max_events)

    res_h_mag = res_h_logfs_norm[start_idx:stop_idx]


    fig, ax = PlotNewResVsFreq(
        fas_freq[:cutoff_idx],
        res_h_mag[..., :cutoff_idx],
        x_label='Freq (Hz)',
        y_label='Residuals',
        flag_xlog=True,
    )
    ax.set_xlim(0.1, 0.96)
    ax.set_ylim(-2, 2)
    ax.set_title(f'FAS Residuals - {mag_label}', fontsize=16)
    fig.tight_layout()




#%% ## Plot spatial residuals (horizontal) for selected frequencies

freq_targets = [0.25, 0.5, 0.75, 0.958]
freq_indices = [int(np.argmin(np.abs(fas_freq - f))) for f in freq_targets]

# One figure with 4 subplots for M4.4
mag_label = 'M4.4'
mag_slice = slice(200, 300)

# Consistent color scale across the 4 panels
spatial_maps = []
for f_idx in freq_indices:
    spatial = np.mean(res_h_logfs_norm[mag_slice, :, f_idx], axis=0).reshape(256, 128)
    spatial_maps.append(spatial)

vmin = -2
vmax = 2
#vmin = min(np.min(m) for m in spatial_maps)
#vmax = max(np.max(m) for m in spatial_maps)

nrows, ncols = 1, 4
fig, axes = plt.subplots(nrows, ncols, figsize=(8, 4), constrained_layout=False)
axes = axes.ravel()
extent = [0.0, 40.0, 0.0, 80.0]

for i, (ax, f_target, f_idx) in enumerate(zip(axes, freq_targets, freq_indices)):
    spatial = np.mean(res_h_logfs_norm[mag_slice, :, f_idx], axis=0).reshape(256, 128)
    im = ax.imshow(
        spatial,
        cmap='coolwarm',
        vmin=vmin,
        vmax=vmax,
        origin='lower',
        aspect='equal',
        extent=extent,
    )
    ax.set_title(f'{mag_label} | {f_target:.2f} Hz', fontsize=11)
    ax.set_xlabel('X (km)', fontsize=11)
    ax.set_xticks(ticks=[10, 20, 30, 40])
    ax.tick_params(axis='x', labelsize=11)

    col = i % ncols
    if col == 0:
        ax.set_ylabel('Y (km)', fontsize=11)
        ax.set_yticks(ticks=[20, 40, 60, 80])
        ax.tick_params(axis='y', labelsize=11)
    else:
        ax.set_ylabel('')
        ax.set_yticklabels([])
        ax.yaxis.set_visible(False)

# colorbar aligned to the last panel
cax = inset_axes(
    axes[-1],
    width='6%',
    height='100%',
    loc='lower left',
    bbox_to_anchor=(1.04, 0.0, 1.0, 1.0),
    bbox_transform=axes[-1].transAxes,
    borderpad=0.0,
)
fig.colorbar(im, cax=cax)

fig.suptitle('Spatial Mean of FAS residual for M.4.4', fontsize=18)
fig.subplots_adjust(left=0.06, right=0.92, top=0.9, bottom=0.12, wspace=0.08)
fig.tight_layout()
# %%
# One figure with 4 subplots for Mw 7
mag_label = 'Mw 7'
mag_slice = slice(100, 200)

# Consistent color scale across the 4 panels
spatial_maps = []
for f_idx in freq_indices:
    spatial = np.mean(res_h_logfs_norm[mag_slice, :, f_idx], axis=0).reshape(256, 128)
    spatial_maps.append(spatial)

vmin = -2
vmax = 2
#vmin = min(np.min(m) for m in spatial_maps)
#vmax = max(np.max(m) for m in spatial_maps)

nrows, ncols = 1, 4
fig, axes = plt.subplots(nrows, ncols, figsize=(8, 4), constrained_layout=False)
axes = axes.ravel()
extent = [0.0, 40.0, 0.0, 80.0]

for i, (ax, f_target, f_idx) in enumerate(zip(axes, freq_targets, freq_indices)):
    spatial = np.mean(res_h_logfs_norm[mag_slice, :, f_idx], axis=0).reshape(256, 128)
    im = ax.imshow(
        spatial,
        cmap='coolwarm',
        vmin=vmin,
        vmax=vmax,
        origin='lower',
        aspect='equal',
        extent=extent,
    )
    ax.set_title(f'{mag_label} | {f_target:.2f} Hz', fontsize=11)
    ax.set_xlabel('X (km)', fontsize=11)
    ax.set_xticks(ticks=[10, 20, 30, 40])
    ax.tick_params(axis='x', labelsize=11)

    col = i % ncols
    if col == 0:
        ax.set_ylabel('Y (km)', fontsize=11)
        ax.set_yticks(ticks=[20, 40, 60, 80])
        ax.tick_params(axis='y', labelsize=11)
    else:
        ax.set_ylabel('')
        ax.set_yticklabels([])
        ax.yaxis.set_visible(False)

# colorbar aligned to the last panel
cax = inset_axes(
    axes[-1],
    width='6%',
    height='100%',
    loc='lower left',
    bbox_to_anchor=(1.04, 0.0, 1.0, 1.0),
    bbox_transform=axes[-1].transAxes,
    borderpad=0.0,
)
fig.colorbar(im, cax=cax)

fig.suptitle('Spatial Mean of FAS residual for Mw 7', fontsize=18)
fig.subplots_adjust(left=0.06, right=0.92, top=0.9, bottom=0.12, wspace=0.08)
fig.tight_layout()
# %%
# One figure with 4 subplots for Mw 6
mag_label = 'Mw 6'
mag_slice = slice(0, 100)

# Consistent color scale across the 4 panels
spatial_maps = []
for f_idx in freq_indices:
    spatial = np.mean(res_h_logfs_norm[mag_slice, :, f_idx], axis=0).reshape(256, 128)
    spatial_maps.append(spatial)

vmin = -2
vmax = 2
#vmin = min(np.min(m) for m in spatial_maps)
#vmax = max(np.max(m) for m in spatial_maps)

nrows, ncols = 1, 4
fig, axes = plt.subplots(nrows, ncols, figsize=(8, 4), constrained_layout=False)
axes = axes.ravel()
extent = [0.0, 40.0, 0.0, 80.0]

for i, (ax, f_target, f_idx) in enumerate(zip(axes, freq_targets, freq_indices)):
    spatial = np.mean(res_h_logfs_norm[mag_slice, :, f_idx], axis=0).reshape(256, 128)
    im = ax.imshow(
        spatial,
        cmap='coolwarm',
        vmin=vmin,
        vmax=vmax,
        origin='lower',
        aspect='equal',
        extent=extent,
    )
    ax.set_title(f'{mag_label} | {f_target:.2f} Hz', fontsize=11)
    ax.set_xlabel('X (km)', fontsize=11)
    ax.set_xticks(ticks=[10, 20, 30, 40])
    ax.tick_params(axis='x', labelsize=11)

    col = i % ncols
    if col == 0:
        ax.set_ylabel('Y (km)', fontsize=11)
        ax.set_yticks(ticks=[20, 40, 60, 80])
        ax.tick_params(axis='y', labelsize=11)
    else:
        ax.set_ylabel('')
        ax.set_yticklabels([])
        ax.yaxis.set_visible(False)

# colorbar aligned to the last panel
cax = inset_axes(
    axes[-1],
    width='6%',
    height='100%',
    loc='lower left',
    bbox_to_anchor=(1.04, 0.0, 1.0, 1.0),
    bbox_transform=axes[-1].transAxes,
    borderpad=0.0,
)
fig.colorbar(im, cax=cax)

fig.suptitle('Spatial Mean of FAS residual for Mw 6', fontsize=18)
fig.subplots_adjust(left=0.06, right=0.92, top=0.9, bottom=0.12, wspace=0.08)
fig.tight_layout()
# # %% FAS all magnitude plot
# import torch
# from typing import Tuple


# def cal_gmean_gstd(wfs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Calculate geometric mean and geometric std along dim=0.
#     """
#     log_wfs = torch.log(wfs)
#     mean = torch.mean(log_wfs, dim=0)
#     std = torch.std(log_wfs, dim=0)
#     return torch.exp(mean).cpu(), torch.exp(std).cpu()


# # validation/FS/H_FS: obs_h_fs shape = [300, 32768, 49]
# # Use 100 events per magnitude, flatten [events, stations, freq] -> [events*stations, freq]
# mag_slices = [
#     ('Mw 4.4', slice(0, 100)),
#     ('Mw 6', slice(100, 200)),
#     ('Mw 7', slice(200, 300)),
# ]

# fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True, constrained_layout=True)
# freq_plot = fas_freq[:49]
# eps = 1e-12

# for ax, (mag_label, mag_slice) in zip(axes, mag_slices):
#     fas_mag = obs_h_fs[mag_slice, :, :49].reshape(-1, 49)
#     fas_mag_tensor = torch.from_numpy(np.clip(fas_mag, eps, None)).float()
#     gmean, gstd = cal_gmean_gstd(fas_mag_tensor)

#     gmean_np = gmean.numpy()
#     gstd_np = gstd.numpy()
#     lower = gmean_np / gstd_np
#     upper = gmean_np * gstd_np

#     ax.plot(freq_plot, gmean_np, color='tab:blue', linewidth=2.0, label='Geometric mean')
#     ax.fill_between(
#         freq_plot,
#         lower,
#         upper,
#         color='tab:blue',
#         alpha=0.25,
#         label='Geometric std band',
#     )
#     ax.set_title(mag_label, fontsize=12)
#     ax.set_xlabel('Frequency (Hz)', fontsize=11)
#     ax.set_xscale('log')
#     ax.set_yscale('log')
#     ax.grid(which='both', alpha=0.35)
#     ax.tick_params(axis='both', labelsize=10)

# axes[0].set_ylabel('FAS', fontsize=11)
# axes[-1].legend(loc='best', fontsize=10)
# fig.suptitle('Validation H_FS FAS Geometric Mean/Std by Magnitude', fontsize=14)

# %%
