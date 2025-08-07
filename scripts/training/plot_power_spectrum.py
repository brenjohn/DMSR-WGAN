#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 16:42:02 2024

@author: brennan
"""

import re
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from dm_analysis import power_spectrum, cloud_in_cells
from swift_tools.fields import get_positions


def plot_spectra(
        lr_data, sr_data, hr_data,
        lr_mass, hr_mass, lr_box_size, hr_box_size, lr_grid_size, hr_grid_size,
        epoch, plots_dir
    ):

    lr_positions = get_positions(lr_data, lr_box_size, lr_grid_size, False)
    lr_density = cloud_in_cells(
        lr_positions.T, lr_grid_size, lr_box_size, periodic=False
    )
    lr_ks, lr_spectrum, lr_uncertainty = power_spectrum(
        lr_density * lr_mass, lr_box_size, lr_grid_size
    )

    sr_positions = get_positions(sr_data, hr_box_size, hr_grid_size, False)
    sr_density = cloud_in_cells(
        sr_positions.T, hr_grid_size, hr_box_size, periodic=False
    )
    sr_ks, sr_spectrum, sr_uncertainty = power_spectrum(
        sr_density * hr_mass, hr_box_size, hr_grid_size
    )

    hr_positions = get_positions(hr_data, hr_box_size, hr_grid_size, False)
    hr_density = cloud_in_cells(
        hr_positions.T, hr_grid_size, hr_box_size, periodic=False
    )
    hr_ks, hr_spectrum, hr_uncertainty = power_spectrum(
        hr_density * hr_mass, hr_box_size, hr_grid_size
    )
    
    plt.figure(figsize=(8, 8))
    plt.plot(hr_ks, hr_spectrum, label='HR', linewidth=4, color='red')
    plt.plot(lr_ks, lr_spectrum, label='LR', linewidth=4)
    plt.plot(sr_ks, sr_spectrum, label='SR', linewidth=4, color='black')
    plt.xscale('log')
    plt.yscale('log')
    # plt.ylim((1e4, 5e7))
    plt.xlabel('k')
    plt.ylabel('P(k)')
    plt.title(f'Epoch {epoch}')
    plt.grid()
    plt.legend()
    
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / f'power_sprectrum_epoch_{epoch:04}.png'
    plt.savefig(plot_path, dpi=100)
    plt.close()


def extract_epoch(filename):
    """Extract epoch number from a sample or plot filename."""
    return int(re.split(r'[._]+', Path(filename).stem)[-1])


#%%
data_dir = Path('./nn_run_c/')
plots_dir = data_dir / 'plots/spectra_191'
samples_dir = data_dir / 'samples_191'

sr_sample_paths = sorted(samples_dir.glob('sr_sample_*.npy'))

# Filter out already plotted samples
existing_plots = [
    extract_epoch(p) for p in plots_dir.glob('power_sprectrum_epoch_*.png')
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
    
    plot_spectra(
        lr_sample, 
        sr_sample, 
        hr_sample,
        LR_mass, 
        HR_mass, 
        LR_patch_length, 
        HR_patch_length,
        LR_patch_size, 
        HR_patch_size,
        epoch,
        plots_dir
    )