#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 16:55:04 2025

@author: brennan
"""

import re
import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from dmsr.field_analysis import nn_distance_field


def plot_nn_distance(
        sr_sample, 
        hr_sample,
        lr_box_size,
        hr_box_size,
        epoch,
        plots_dir = 'plots/training_samples/',
        save=True
    ):
    sr_data = torch.from_numpy(sr_sample)
    hr_data = torch.from_numpy(hr_sample)
    
    sr_df = nn_distance_field(sr_data, hr_box_size)
    hr_df = nn_distance_field(hr_data, hr_box_size)
    
    sr_dx, sr_dy, sr_dz = project(sr_df)
    hr_dx, hr_dy, hr_dz = project(hr_df)
    
    vmax = max(hr_dx.max(), hr_dy.max(), hr_dz.max())
    vmin = min(hr_dx.min(), hr_dy.min(), hr_dz.min())
    
    # Create a figure
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # LR scatter plot
    axes[0, 0].imshow(sr_dx, vmax=vmax, vmin=vmin)
    axes[0, 1].imshow(sr_dy, vmax=vmax, vmin=vmin)
    axes[0, 2].imshow(sr_dz, vmax=vmax, vmin=vmin)
    axes[0, 0].set_ylabel('SR', fontsize=20)
    
    # LR scatter plot
    axes[1, 0].imshow(hr_dx, vmax=vmax, vmin=vmin)
    axes[1, 1].imshow(hr_dy, vmax=vmax, vmin=vmin)
    axes[1, 2].imshow(hr_dz, vmax=vmax, vmin=vmin)
    axes[1, 0].set_ylabel('HR', fontsize=20)
    
    for a in axes.flatten():
        a.set_xticks([])
        a.set_yticks([])
    
    # Add title and adjust layout
    fig.suptitle(f'Epoch {epoch}')
    plt.tight_layout()
    # plt.show()
    
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / f'nn_distance_epoch_{epoch:04}.png'
    fig.savefig(plot_path, dpi=100)
    plt.close(fig)


def project(dn):
    dn = -1/(10*dn + 0.01)
    # dn = torch.log(100*dn)
    dn = torch.sum(dn, dim=-1)
    # print(dn.max(), dn.min())
    
    dx = dn[0, 0, ...].flip(dims=(0,))
    dy = dn[0, 2, ...].flip(dims=(0,))
    dz = dn[0, 4, ...].flip(dims=(0,))
    
    return dx, dy, dz


def extract_epoch(filename):
    """Extract epoch number from a sample or plot filename."""
    return int(re.split(r'[._]+', Path(filename).stem)[-1])


#%%
data_dir = Path('./nn_run_c/')
plots_dir = data_dir / 'plots/nn_distance_191'
samples_dir = data_dir / 'samples_191'

sr_sample_paths = sorted(samples_dir.glob('sr_sample_*.npy'))

# Filter out already plotted samples
existing_plots = [
    extract_epoch(p) for p in plots_dir.glob('nn_distance_epoch_*.png')
]
sr_sample_paths = [
    p for p in sr_sample_paths if extract_epoch(p) not in existing_plots
]

# Load metadata
metadata = np.load(data_dir / 'metadata.npy')
box_size        = metadata[0]
LR_patch_length = metadata[1]
HR_patch_length = metadata[2]
LR_patch_size   = int(metadata[3])
HR_patch_size   = int(metadata[4])
LR_inner_size   = int(metadata[5])
padding         = int(metadata[6])
LR_mass         = metadata[7]
HR_mass         = metadata[8]

# Load normalisation stats
training_summary_stats = np.load(
    data_dir / 'normalisation.npy', allow_pickle=True
).item()
lr_std = training_summary_stats['LR_disp_fields_std']
hr_std = training_summary_stats['HR_disp_fields_std']

# Load reference samples
lr_sample = lr_std * np.load(samples_dir / 'lr_sample.npy')[:, :3, ...]
hr_sample = hr_std * np.load(samples_dir / 'hr_sample.npy')[:, :3, ...]


# Plot loop
for sr_path in sr_sample_paths:
    epoch = extract_epoch(sr_path)
    sr_sample = hr_std * np.load(sr_path)[:, :3, ...]
    
    plot_nn_distance(
        sr_sample, 
        hr_sample, 
        LR_patch_length, 
        HR_patch_length, 
        epoch,
        plots_dir
    )