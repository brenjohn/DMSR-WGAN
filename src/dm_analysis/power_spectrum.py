#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 11:20:10 2025

@author: brennan
"""

import numpy as np

from numpy.fft import fftn, fftfreq


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
    # Get the fourier transform of the density field.
    density_ft = fftn(density) / (grid_size**3)
    power_spectrum_k = np.abs(density_ft)**2 * box_size**3 
    
    # Compute the frequency arrays
    ks = 2 * np.pi * fftfreq(grid_size, box_size/grid_size)
    kx, ky, kz = np.meshgrid(ks, ks, ks, indexing='ij')
    k = np.sqrt(kx**2 + ky**2 + kz**2)
    
    # Radial bins
    k_bins = np.linspace(0, np.max(k), grid_size//2)
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