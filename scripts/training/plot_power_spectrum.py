#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 16:42:02 2024

@author: brennan
"""

import sys
sys.path.append("..")
sys.path.append("../..")

import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt

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
    
    os.makedirs(plots_dir, exist_ok=True)
    plot_name = plots_dir + f'power_sprectrum_epoch_{epoch:04}.png'
    plt.savefig(plot_name, dpi=100)  
    plt.close()


#%%
data_dir = './z_run/'
plots_dir = data_dir + 'plots/spectra_191/'
samples_dir = data_dir + 'samples_191/'
sr_samples = glob.glob(samples_dir + 'sr_sample_*.npy')
sr_samples = np.sort(sr_samples)

existing_plots = glob.glob(plots_dir + 'power_sprectrum_epoch_*.png')
existing_plots = [re.split(r'[._]+', plot)[-2] for plot in existing_plots]
sr_samples = [sample for sample in sr_samples 
              if not re.split(r'[._]+', sample)[-2] in existing_plots]

metadata = np.load(data_dir + 'metadata.npy')
box_size        = metadata[0]
LR_patch_length = metadata[1]
HR_patch_length = metadata[2]
LR_patch_size   = int(metadata[3])
HR_patch_size   = int(metadata[4])
LR_inner_size   = int(metadata[5])
padding         = int(metadata[6])
LR_mass         = metadata[7]
HR_mass         = metadata[8]

training_summary_stats = np.load(
    data_dir + 'normalisation.npy', allow_pickle=True
).item()
lr_std = training_summary_stats['LR_disp_fields_std']
hr_std = training_summary_stats['HR_disp_fields_std']

lr_sample = np.load(samples_dir + 'lr_sample.npy')[:, :3, ...]
hr_sample = np.load(samples_dir + 'hr_sample.npy')[:, :3, ...]


for sr_sample in sr_samples:
    epoch = int(re.split(r'[._]+', sr_sample)[-2])
    sr_sample = np.load(sr_sample)[:, :3, ...]
    
    plot_spectra(
        lr_std * lr_sample, hr_std * sr_sample, hr_std * hr_sample,
        LR_mass, HR_mass, LR_patch_length, HR_patch_length,
        LR_patch_size, HR_patch_size,
        epoch,
        plots_dir
    )