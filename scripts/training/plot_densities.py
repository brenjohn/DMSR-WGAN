#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 11 12:14:14 2025

@author: brennan
"""

import sys
sys.path.append("..")
sys.path.append("../..")

import os
import re
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt

from dm_analysis import power_spectrum, cloud_in_cells
from dmsr.field_analysis import displacements_to_positions
from swift_tools.fields import get_positions


def plot_densities(
        lr_sample, 
        sr_sample, 
        hr_sample,
        lr_box_size,
        hr_box_size,
        lr_grid_size,
        hr_grid_size,
        epoch,
        plots_dir = 'plots/training_samples/',
        save=True
    ):
    
    lr_positions = get_positions(lr_sample, lr_box_size, lr_grid_size, False)
    lr_density = cloud_in_cells(
        lr_positions.T, lr_grid_size, lr_box_size, periodic=False
    )
    lr_density = np.sum(lr_density, axis=-1)
    
    sr_positions = get_positions(sr_sample, hr_box_size, hr_grid_size, False)
    sr_density = cloud_in_cells(
        sr_positions.T, hr_grid_size, hr_box_size, periodic=False
    )
    sr_density = np.sum(sr_density, axis=-1)

    hr_positions = get_positions(hr_sample, hr_box_size, hr_grid_size, False)
    hr_density = cloud_in_cells(
        hr_positions.T, hr_grid_size, hr_box_size, periodic=False
    )
    hr_density = np.sum(hr_density, axis=-1)
    
    # Create a figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 8))
    
    # LR scatter plot
    ax1.imshow(np.flip(lr_density, axis=0))
    ax1.set_title('LR')
    
    # HR scatter plot
    ax3.imshow(np.flip(hr_density, axis=0))
    ax3.set_title('HR')
    
    # SR scatter plot
    ax2.imshow(np.flip(sr_density, axis=0))
    ax2.set_title('SR')
    ax2.set_xlim(ax3.get_xlim())
    ax2.set_ylim(ax3.get_ylim())
    
    # Add title and adjust layout
    fig.suptitle(f'Epoch {epoch}')
    plt.tight_layout()
    
    os.makedirs(plots_dir, exist_ok=True)
    plot_name = plots_dir + f'density_plot_epoch_{epoch:04}.png'
    fig.savefig(plot_name, dpi=100)
    plt.close(fig)


#%%
data_dir = './nn_run/'
plots_dir = data_dir + 'plots/sample_density_191/'
samples_dir = data_dir + 'samples_191/'
sr_samples = glob.glob(samples_dir + 'sr_sample_*.npy')
sr_samples = np.sort(sr_samples)

existing_plots = glob.glob(plots_dir + 'density_plot_epoch_*.png')
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
    
    plot_densities(
        lr_std * lr_sample, hr_std * sr_sample, hr_std * hr_sample, 
        LR_patch_length, HR_patch_length, 
        LR_patch_size, HR_patch_size,
        epoch,
        plots_dir
    )