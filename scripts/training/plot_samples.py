#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 13:35:32 2024

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

from swift_tools.data import load_normalisation_parameters
from dmsr.field_operations.conversion import displacements_to_positions

def plot_samples(
        lr_sample, 
        sr_sample, 
        hr_sample,
        lr_box_size,
        hr_box_size,
        epoch,
        plots_dir = 'plots/training_samples/',
        save=True
    ):
    lr_data = torch.from_numpy(lr_sample)
    sr_data = torch.from_numpy(sr_sample)
    hr_data = torch.from_numpy(hr_sample)
    
    lr_positions = displacements_to_positions(lr_data, lr_box_size)
    sr_positions = displacements_to_positions(sr_data, hr_box_size)
    hr_positions = displacements_to_positions(hr_data, hr_box_size)
    
    lr_xs, lr_ys = get_xys(lr_positions)
    sr_xs, sr_ys = get_xys(sr_positions)
    hr_xs, hr_ys = get_xys(hr_positions)
    
    # Create a figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 8))
    
    # LR scatter plot
    ax1.scatter(lr_xs, lr_ys, alpha=0.2, s=0.5)
    ax1.set_title('LR')
    
    # HR scatter plot
    ax3.scatter(hr_xs, hr_ys, alpha=0.1, s=0.1)
    ax3.set_title('HR')
    
    # SR scatter plot
    ax2.scatter(sr_xs, sr_ys, alpha=0.1, s=0.1)
    ax2.set_title('SR')
    ax2.set_xlim(ax3.get_xlim())
    ax2.set_ylim(ax3.get_ylim())
    
    # Add title and adjust layout
    fig.suptitle(f'Epoch {epoch}')
    plt.tight_layout()
    
    os.makedirs(plots_dir, exist_ok=True)
    plot_name = plots_dir + f'particle_plot_epoch_{epoch:04}.png'
    fig.savefig(plot_name, dpi=100)
    plt.close(fig)

    
def get_xys(positions):
    positions = torch.transpose(positions, 1, -1)
    positions = positions.reshape((-1, 3))
    xs = positions[:, 0]
    ys = positions[:, 1]
    return xs, ys


#%%
data_dir = './velocity_run/'
plots_dir = data_dir + 'plots/training_samples/'
samples_dir = data_dir + 'samples/'
sr_samples = glob.glob(samples_dir + 'sr_sample_*.npy')
sr_samples = np.sort(sr_samples)

existing_plots = glob.glob(plots_dir + 'particle_plot_epoch_*.png')
existing_plots = [re.split(r'[._]+', plot)[-2] for plot in existing_plots]
sr_samples = [sample for sample in sr_samples 
              if not re.split(r'[._]+', sample)[-2] in existing_plots]

scale_path = data_dir + 'normalisation.npy'    
lr_std, hr_std, _, _ = load_normalisation_parameters(scale_path)

lr_sample = np.load(samples_dir + 'lr_sample.npy')[:, :3, ...] * lr_std
hr_sample = np.load(samples_dir + 'hr_sample.npy')[:, :3, ...] * hr_std

# TODO: read this from metadata
box_size = 35.56187768431281

for sr_sample in sr_samples:
    epoch = int(re.split(r'[._]+', sr_sample)[-2])
    sr_sample = np.load(sr_sample)[:, :3, ...] * hr_std
    
    plot_samples(
        lr_sample, sr_sample, hr_sample, 
        20*box_size/16, box_size, epoch,
        plots_dir
    )