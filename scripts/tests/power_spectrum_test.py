#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 11:15:19 2024

@author: brennan
"""

import sys
sys.path.append("..")
sys.path.append("../..")

import torch
import matplotlib.pyplot as plt

from dmsr.data_tools import load_numpy_dataset
from dmsr.data_tools import DMSRDataset
from dmsr.field_operations.conversion import cic_density_field
from dmsr.analysis import power_spectrum


def compute_power_spectrum(displacements, particle_mass, box_size, grid_size):
    cell_size = box_size / grid_size
    cell_volume = cell_size**3
    
    # Compute the denisty field from the given displacement field.
    density = cic_density_field(displacements[None, :3, ...], box_size).numpy()
    density = density[0, 0, ...] * particle_mass / cell_volume
    density = torch.from_numpy(density)
    
    return power_spectrum(density, box_size, grid_size)



#%%
data_directory = '../../data/dmsr_training_velocity_x64/'
data = load_numpy_dataset(data_directory)
LR_data, HR_data, box_size, LR_grid_size, HR_grid_size = data

n = 0
LR_data, HR_data = LR_data[n:n+1, ...], HR_data[n:n+1, ...]

dataset = DMSRDataset(LR_data, HR_data, augment=False)
lr_sample, hr_sample = dataset[0]


#%%
lr_bins, lr_power_spectrum, lr_uncertainty = compute_power_spectrum(
    lr_sample, 64, 20*box_size/16, lr_sample.shape[-1]
)
hr_bins, hr_power_spectrum, hr_uncertainty = compute_power_spectrum(
    hr_sample, 1, box_size, hr_sample.shape[-1]
)

plt.errorbar(hr_bins, hr_power_spectrum, yerr=hr_uncertainty, fmt='-o')
plt.errorbar(lr_bins, lr_power_spectrum, yerr=lr_uncertainty, fmt='-o')
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'Delta**2')
plt.xlabel('k')
plt.show()
