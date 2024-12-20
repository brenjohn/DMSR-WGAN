#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 11:29:19 2024

@author: brennan

This file defines functions functions for analysing some tensors representing
physical quantities.
"""

import torch

from torch.fft import fftn, fftfreq
from .field_operations.conversion import cic_density_field


def power_spectrum(density, box_size, grid_size):
    """Compute the dimensionless power spectrum of the given denisty field.

    Parameters
    ----------
    density : numpy array
        An array conatining a denisty field to compute the power spectrum for.
    box_size : int
        The side length of the box containing the density field.
    grid_size : int
        The number of cells along an edge of the box.

    Returns
    -------
    k_bin_centers : numpy array
        The centre points of the k-bins used to compute the power spectrum.
    power_spectrum : numpy array
        The dimensionless power spectrum for the given denisty field.
    uncertainty : numpy array
        The uncertainty of the computed power spectrum.
    """
    device = density.device
    
    # Get the fourier transform of the density field.
    density_ft = fftn(density) / (grid_size**3)
    power_spectrum_k = torch.abs(density_ft)**2 * box_size**3 
    
    # Compute the frequency arrays
    ks = 2 * torch.pi * fftfreq(grid_size, box_size/grid_size, device=device)
    kx, ky, kz = torch.meshgrid(ks, ks, ks, indexing='ij')
    k = torch.sqrt(kx**2 + ky**2 + kz**2)
    
    # Radial bins
    k_bins = torch.linspace(0, torch.max(k), steps=grid_size//2, device=device)
    k_bin_centers = 0.5 * (k_bins[1:] + k_bins[:-1])
    
    # Average the power spectrum over spherical shells
    power_spectrum = torch.zeros_like(k_bin_centers)
    uncertainty = torch.zeros_like(k_bin_centers)
    
    for i in range(len(k_bin_centers)):
        shell_mask = (k >= k_bins[i]) & (k < k_bins[i+1])
        power = power_spectrum_k[shell_mask] 
        power *= k[shell_mask]**3 / (2 * torch.pi**2)
        
        power_spectrum[i] = torch.mean(power)
        uncertainty[i] = power_spectrum[i] / torch.sqrt(torch.sum(shell_mask))
    
    return k_bin_centers, power_spectrum, uncertainty


def displacement_power_spectrum(
        displacements, particle_mass, box_size, grid_size
    ):
    """Compute the power spectrum of the cic density field constructed from the
    given displacement field.
    """
    cell_size = box_size / grid_size
    cell_volume = cell_size**3
    
    # Compute the denisty field from the given displacement field.
    density = cic_density_field(displacements, box_size, grid_size)
    density = density[0, 0, ...] * particle_mass / cell_volume
    
    return power_spectrum(density, box_size, grid_size)