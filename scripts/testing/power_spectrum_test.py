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
import numpy as np
import matplotlib.pyplot as plt

from numpy.fft import fftn

from dmsr.swift_processing import load_numpy_dataset
from dmsr.dmsr_gan.dmsr_dataset import DMSRDataset
from dmsr.field_operations.conversion import cic_density_field


#%%

def compute_power_spectrum(displacements, particle_mass, box_size, grid_size):
    cell_size = box_size / grid_size
    cell_volume = cell_size**3
    
    # Compute the denisty field from the given displacement field.
    density = cic_density_field(displacements[None, ...], box_size).numpy()
    density = density[0, 0, ...] * particle_mass / cell_volume
    
    return power_spectrum(density, box_size, grid_size)


def power_spectrum(density, box_size, grid_size):
    # Get the fourier transform of the density field.
    density_ft = fftn(density) / (grid_size**3)
    power_spectrum_k = np.abs(density_ft)**2 * box_size**3 
    
    # Compute the frequency arrays
    ks = 2 * np.pi * np.fft.fftfreq(grid_size, box_size/grid_size)
    kx, ky, kz = np.meshgrid(ks, ks, ks, indexing='ij')
    k = np.sqrt(kx**2 + ky**2 + kz**2)
    
    # Radial bins
    k_bins = np.linspace(0, np.max(k), num=grid_size//2)
    k_bin_centers = 0.5 * (k_bins[1:] + k_bins[:-1])
    
    # Average the power spectrum over spherical shells
    power_spectrum = np.zeros_like(k_bin_centers)
    uncertainty = np.zeros_like(k_bin_centers)
    for i in range(len(k_bin_centers)):
        shell_mask = (k >= k_bins[i]) & (k < k_bins[i+1])
        power = power_spectrum_k[shell_mask] 
        power *= k[shell_mask]**3 / (2 * np.pi**2)
        
        power_spectrum[i] = np.mean(power)
        uncertainty[i] = power_spectrum[i] / np.sqrt(np.sum(shell_mask))
    
    return k_bin_centers, power_spectrum, uncertainty



#%%

data_directory = '../../data/dmsr_training/'
data = load_numpy_dataset(data_directory)
LR_data, HR_data, box_size, LR_grid_size, HR_grid_size = data

n = 0
LR_data, HR_data = LR_data[n:n+1, ...], HR_data[n:n+1, ...]

dataset = DMSRDataset(LR_data, HR_data, augment=False)


#%%
    
lr_sample, hr_sample = dataset[0]


#%%

lr_bins, lr_power_spectrum, lr_uncertainty = compute_power_spectrum(lr_sample, 8, 20*box_size/16, 20)
hr_bins, hr_power_spectrum, hr_uncertainty = compute_power_spectrum(hr_sample, 1, box_size, 32)

plt.errorbar(hr_bins, hr_power_spectrum, yerr=hr_uncertainty, fmt='-o')
plt.errorbar(lr_bins, lr_power_spectrum, yerr=lr_uncertainty, fmt='-o')
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'Delta**2')
plt.xlabel('k')
plt.show()
