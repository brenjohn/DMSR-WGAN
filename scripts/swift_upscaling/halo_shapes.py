#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 23:31:43 2025

@author: john

This script was used to create figure 7 of Brennan et. al. 2025 "On the Use of
WGANs for Super Resolution in Dark-Matter Simulations".
"""

import sys
sys.path.append("..")
sys.path.append("../..")

import numpy as np
import matplotlib.pyplot as plt

from swift_tools.data import read_snapshot
from swift_tools.friends_of_friends import friends_of_friends
from swift_tools.friends_of_friends import Halo, compute_shape


#%% Read in snapshot HR and SR snapshots.
data_dir = './swift_snapshots/'
hr_snapshot = data_dir + '128/snap_0002.hdf5'
sr_snapshot = data_dir + '064/snap_0002_sr_level_0.hdf5'

hr_positions, _, hr_box_size, h, hr_mass = read_snapshot(hr_snapshot)
sr_positions, _, sr_box_size, h, sr_mass = read_snapshot(sr_snapshot)


#%% Compute halo catalogues for HR and SR data.
def halo_catalogue(positions, box_size, mass):
    """Returns a list of masses for all friends-of-friends halos.
    """
    b = 0.2 # Sets the linking length fraction.
    N = 100 # Only consider halos with at least this many particles.
    volume = box_size**3
    mean_interparticle_separation = (volume / len(positions))**(1/3)
    linking_length = b * mean_interparticle_separation
    halos = friends_of_friends(positions, box_size, linking_length)
    return [
        Halo.from_halo_dict(halo_dict, mass) 
        for halo_dict in halos if len(halo_dict) > N
    ]

print('Getting hr halos')
hr_halos = halo_catalogue(hr_positions, hr_box_size, hr_mass * 1e10)

print('Getting sr halos')
sr_halos = halo_catalogue(sr_positions, sr_box_size, sr_mass * 1e10)


#%% Compute the shapes for halos in the HR and SR halo catalogues.
def get_shapes(halos, particle_mass):
    masses = []
    bas = []
    cas = []
    for i, halo in enumerate(halos):
        print('Halo', i)
        ratios = compute_shape(halo, 35)
        if ratios is not None:
            ba, ca = ratios
            bas.append(ba)
            cas.append(ca)
            masses.append(halo.mass)
    return masses, bas, cas

hr_masses, hr_bas, hr_cas = get_shapes(hr_halos, hr_mass)
sr_masses, sr_bas, sr_cas = get_shapes(sr_halos, sr_mass)


#%% Scatter plot of halo mass vs b/a shape parameter.
plt.scatter(hr_masses, hr_bas, label='HR')
plt.scatter(sr_masses, sr_bas, label='SR')
plt.xscale('log')
plt.xlabel('Mass', fontsize=14)
plt.ylabel('b / a', fontsize=14)
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig('ba.png', dpi=210)
plt.show()
plt.close()


#%% Scatter plot of halo mass vs c/a shape parameter.
plt.scatter(hr_masses, hr_cas, label='HR')
plt.scatter(sr_masses, sr_cas, label='SR', alpha=1)
plt.xscale('log')
plt.xlabel('Mass', fontsize=14)
plt.ylabel('c / a', fontsize=14)
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig('ca_level_1.png', dpi=210)
plt.show()
plt.close()


#%% Plot of halo mass vs median shape parameter.
bins = np.logspace(12, 16, 5)
hr_bin_indices = np.digitize(hr_masses, bins)
sr_bin_indices = np.digitize(sr_masses, bins)

hr_means = []
sr_means = []
hr_bin_centers = []
sr_bin_centers = []

for i in range(1, len(bins)):
    in_bin = (hr_bin_indices == i)
    if np.any(in_bin):
        median_shape = np.median(np.asarray(hr_cas)[in_bin])
        hr_means.append(median_shape)
        hr_bin_centers.append((bins[i] + bins[i - 1]) / 2)
        
    in_bin = (sr_bin_indices == i)
    if np.any(in_bin):
        median_shape = np.median(np.asarray(sr_cas)[in_bin])
        sr_means.append(median_shape)
        sr_bin_centers.append((bins[i] + bins[i - 1]) / 2)
        
plt.plot(hr_bin_centers, hr_means, lw=2, label='Median HR shape')
plt.plot(sr_bin_centers, sr_means, lw=2, label='Median SR shape')
plt.xscale('log')
plt.ylim(0, 1)
plt.xlabel('Mass', fontsize=14)
plt.ylabel(r'$c/a$', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig('median_shape_level_1.png', dpi=210)
plt.show()
plt.close()


#%% Scatter plot of haloe shape parameters b/a vs c/a.
plt.scatter(hr_bas, hr_cas, label='HR', s=np.log(hr_masses))
plt.scatter(sr_bas, sr_cas, label='SR', s=np.log(sr_masses))
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
# plt.xscale('log')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('b / a')
plt.ylabel('c / a')
plt.legend()
plt.tight_layout()
plt.savefig('ba_vs_ca_level_1.png', dpi=210)
plt.show()
plt.close()


#%% Plot of distribution of shape parameters (b/a, c/a) for HR and SR halos.
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

def shifted_cmap(base_cmap, ncolors=256, start_white_frac=0.14):
    base = plt.get_cmap(base_cmap)
    colors = base(np.linspace(0, 1, ncolors))
    colors[:int(ncolors * start_white_frac)] = [1, 1, 1, 1]
    return ListedColormap(colors)

def get_sigma_levels(kde, grid_coords, levels=[0.85, 0.15]):
    """
    Compute the density thresholds corresponding to given cumulative levels
    for a 2D KDE.
    """
    # Evaluate the KDE on the grid
    density = kde(grid_coords)
    density_sorted = np.sort(density)[::-1]  # sort descending
    cumsum = np.cumsum(density_sorted)
    cumsum /= cumsum[-1]  # normalize to 1

    # Find density thresholds for each level
    thresholds = [
        density_sorted[np.searchsorted(cumsum, level)] for level in levels
    ]
    return thresholds


# Prepare Data
data1 = np.stack([np.asarray(hr_bas), np.asarray(hr_cas)])
data2 = np.stack([np.asarray(sr_bas), np.asarray(sr_cas)])

# Compute Gaussian kernel density estimation of shape parameter PDFs..
X, Y = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
grid_coords = np.vstack([X.ravel(), Y.ravel()])
kde1 = gaussian_kde(data1)
kde2 = gaussian_kde(data2)
Z1 = kde1(grid_coords).reshape(X.shape)
Z2 = kde2(grid_coords).reshape(X.shape)

# Plotting
fig = plt.figure(figsize=(8, 8))
grid = plt.GridSpec(4, 4, hspace=0.05, wspace=0.05)

# Main contour plot.
ax_main = fig.add_subplot(grid[1:, :-1])
ax_main.plot([0, 1], [0, 1], color='black', linestyle='--')
ax_main.set_xlim(0, 1)
ax_main.set_ylim(0, 1)
sigma_levels1 = get_sigma_levels(kde1, grid_coords)
sigma_levels2 = get_sigma_levels(kde2, grid_coords)
contour1 = ax_main.contourf(
    X, Y, Z1, 
    levels=sigma_levels1 + [Z1.max()], cmap=shifted_cmap("Reds"), alpha=0.5,
)
contour2 = ax_main.contourf(
    X, Y, Z2, 
    levels=sigma_levels2 + [Z2.max()], cmap=shifted_cmap("Blues"), alpha=0.5,
)
contour_lines1 = ax_main.contour(
    X, Y, Z1, levels=sigma_levels1, colors='red', linewidths=2
)
contour_lines2 = ax_main.contour(
    X, Y, Z2, levels=sigma_levels2, colors='blue', linewidths=2
)

# Labels
ax_main.set_xlabel(r'$b / a$', fontsize=21)
ax_main.set_ylabel(r'$c / a$', fontsize=21)
ax_main.tick_params(axis='both', labelsize=16)
ax_main.set_xticks([0.2, 0.4, 0.6, 0.8])
ax_main.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])

# Legend with matching colors
hr_patch = mpatches.Patch(color='red', alpha=0.6, label='HR')
sr_patch = mpatches.Patch(color='blue', alpha=0.6, label='SR')
ax_main.legend(handles=[hr_patch, sr_patch], fontsize=21)
ax_main.grid()

# Marginal plots for b/a
xs = np.arange(0, 1, 0.002)
ax_bas = fig.add_subplot(grid[0, :-1], sharex=ax_main)
hr_kde = gaussian_kde(np.asarray(hr_bas))
sr_kde = gaussian_kde(np.asarray(sr_bas))
ax_bas.plot(xs, hr_kde(xs), color='red', linewidth=2)
ax_bas.plot(xs, sr_kde(xs), color='blue', linewidth=2)
ax_bas.set_ylim(0, 3)
ax_bas.set_yticks([0, 2])
ax_bas.set_ylabel('b/a density', fontsize=21)
ax_bas.tick_params(axis='x', labelsize=0)
ax_bas.tick_params(axis='y', labelsize=16)

# Marginal plots for c/a
xs = np.arange(0, 1, 0.002)
ax_bas = fig.add_subplot(grid[1:, -1], sharey=ax_main)
hr_kde = gaussian_kde(np.asarray(hr_cas))
sr_kde = gaussian_kde(np.asarray(sr_cas))
ax_bas.plot(hr_kde(xs), xs, color='red', linewidth=2)
ax_bas.plot(sr_kde(xs), xs, color='blue', linewidth=2)
ax_bas.set_xlim(0, 3)
ax_bas.set_xlabel('c/a density', fontsize=21)
ax_bas.tick_params(axis='x', labelsize=16)
ax_bas.tick_params(axis='y', labelsize=0)

# Show plot
plt.tight_layout()
plt.savefig('SR_vs_HR_shape_distribution.png', dpi=210)
plt.show()