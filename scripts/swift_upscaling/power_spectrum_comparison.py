#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 17:36:56 2024

@author: brennan

This script was used to create figure 4 of Brennan et. al. 2025 "On the Use of
WGANs for Super Resolution in Dark-Matter Simulations".
"""

import sys
sys.path.append("..")
sys.path.append("../..")

import torch
import matplotlib.pyplot as plt

from swift_tools.density import cloud_in_cells
from swift_tools.data import read_snapshot
from dmsr.analysis import power_spectrum


#%% Calculating Comological Volume Density Field.
data_dir = './swift_snapshots/'
lr_snapshot = data_dir + '064/snap_0002.hdf5'
hr_snapshot = data_dir + '128/snap_0002.hdf5'
sr_snapshot = data_dir + '064/snap_0002_sr.hdf5'

lr_positions, lr_grid_size, lr_box_size, h, _ = read_snapshot(lr_snapshot)
hr_positions, hr_grid_size, hr_box_size, h, _ = read_snapshot(hr_snapshot)
sr_positions, hr_grid_size, hr_box_size, h, _ = read_snapshot(sr_snapshot)

lr_density = cloud_in_cells(lr_positions, lr_grid_size, lr_box_size)
hr_density = cloud_in_cells(hr_positions, hr_grid_size, hr_box_size)
sr_density = cloud_in_cells(sr_positions, hr_grid_size, hr_box_size)


#%% Power Spectrum Comparison.
def spectrum(denisty, box_size, grid_size):
    denisty = torch.from_numpy(denisty)
    ks, spectrum, _ = power_spectrum(denisty, box_size, grid_size)
    return ks, spectrum


# Note, lr particles are 8 times more massive.
lr_ks, lr_spectrum = spectrum(lr_density * 8, lr_box_size, lr_grid_size)
hr_ks, hr_spectrum = spectrum(hr_density, hr_box_size, hr_grid_size)
sr_ks, sr_spectrum = spectrum(sr_density, hr_box_size, hr_grid_size)

fig, axes = plt.subplots(
    2, 1, 
    figsize=(6, 5), 
    sharex=True, 
    gridspec_kw={'hspace': 0}
)

axes[0].plot(lr_ks, lr_spectrum, linewidth=4, label='Low Resolution')
axes[0].plot(hr_ks, hr_spectrum, linewidth=4, label='High Resolution')
axes[0].plot(sr_ks, sr_spectrum, linewidth=4, label='Super Resolution')
axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].set_ylabel(r'$\Delta(k)$', fontsize=16)
axes[0].legend(fontsize=14)
axes[0].grid()
axes[0].tick_params(axis='both', which='major', labelsize=14)

axes[1].plot(lr_ks, hr_spectrum[:len(lr_spectrum)] / lr_spectrum, linewidth=4)
axes[1].plot(hr_ks, hr_spectrum / hr_spectrum, linewidth=4)
axes[1].plot(hr_ks, hr_spectrum / sr_spectrum, linewidth=4)
axes[1].set_ylim(0.5, 4)
axes[1].set_xlabel(r'$k$   [Mpc$^{-1}$]', fontsize=16)
axes[1].set_ylabel(r'$\Delta_{HR} / \Delta$', fontsize=16)
axes[1].set_yticks([1, 2, 3])
axes[1].grid()
axes[1].tick_params(axis='both', which='major', labelsize=14)
plt.tight_layout()
plt.savefig('power_spectrum.png', dpi=300)
plt.show()
plt.close()