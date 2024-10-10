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
import torch
import numpy as np
import matplotlib.pyplot as plt

from dmsr.field_operations.conversion import cic_density_field

from numpy.fft import fftn, fftfreq


def compute_power_spectrum(displacements, particle_mass, box_size, grid_size):
    cell_size = box_size / grid_size
    cell_volume = cell_size**3
    
    # Compute the denisty field from the given displacement field.
    density = cic_density_field(displacements, box_size).numpy()
    density = density[0, 0, ...] * particle_mass / cell_volume
    
    return power_spectrum(density, box_size, grid_size)


def power_spectrum(density, box_size, grid_size):
    # Get the fourier transform of the density field.
    density_ft = fftn(density) / (grid_size**3)
    power_spectrum_k = np.abs(density_ft)**2 * box_size**3 
    
    # Compute the frequency arrays
    ks = 2 * np.pi * fftfreq(grid_size, box_size/grid_size)
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


def plot_spectra(
        lr_data, sr_data, hr_data,
        lr_mass, hr_mass, lr_box_size, hr_box_size, lr_grid_size, hr_grid_size,
        epoch, plots_dir):

    lr_ks, lr_spectrum, lr_uncertainty = compute_power_spectrum(
        lr_data, lr_mass, lr_box_size, lr_grid_size
    )

    sr_ks, sr_spectrum, sr_uncertainty = compute_power_spectrum(
        sr_data, hr_mass, hr_box_size, hr_grid_size
    )

    hr_ks, hr_spectrum, hr_uncertainty = compute_power_spectrum(
        hr_data, hr_mass, hr_box_size, hr_grid_size
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
plots_dir = 'plots/training_spectra/'
data_dir = './data/samples/'
sr_samples = glob.glob(data_dir + 'sr_sample_*.npy')
sr_samples = np.sort(sr_samples)

existing_plots = glob.glob(plots_dir + 'power_sprectrum_epoch_*.png')
existing_plots = [re.split(r'[._]+', plot)[-2] for plot in existing_plots]
sr_samples = [sample for sample in sr_samples 
              if not re.split(r'[._]+', sample)[-2] in existing_plots]

lr_sample = np.load(data_dir + 'lr_sample.npy')
lr_sample = torch.from_numpy(lr_sample)
hr_sample = np.load(data_dir + 'hr_sample.npy')
hr_sample = torch.from_numpy(hr_sample)

# TODO: read this from metadata
box_size = 35.56187768431281

for sr_sample in sr_samples:
    epoch = int(re.split(r'[._]+', sr_sample)[-2])
    sr_sample = np.load(sr_sample)
    sr_sample = torch.from_numpy(sr_sample)
    
    plot_spectra(
        lr_sample, sr_sample, hr_sample,
        64, 1, 20*box_size/14, box_size, 20, 56,
        epoch, plots_dir
    )