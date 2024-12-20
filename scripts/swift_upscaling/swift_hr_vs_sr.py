#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 17:36:56 2024

@author: brennan
"""

import sys
sys.path.append("..")
sys.path.append("../..")

import h5py as h5
import numpy as np

from swift_tools.density import cloud_in_cells

import matplotlib.pyplot as plt
import matplotlib.patches as patches


#%% Calculating Comological Volume Density Field.
def read_snapshot(snapshot):
    file = h5.File(snapshot, 'r')
    
    h = file['Cosmology'].attrs['h'][0]
    
    grid_size = file['ICs_parameters'].attrs['Grid Resolution']
    box_size = file['Header'].attrs['BoxSize'][0]
    
    dm_data = file['DMParticles']
    positions = np.asarray(dm_data['Coordinates'])
    
    file.close()
    return positions, grid_size, box_size, h


data_dir = './swift_snapshots/'
lr_snapshot = data_dir + '064/snap_0002.hdf5'
hr_snapshot = data_dir + '128/snap_0002.hdf5'
sr_snapshot = data_dir + '064/snap_0002_sr.hdf5'

lr_positions, lr_grid_size, lr_box_size, h = read_snapshot(lr_snapshot)
hr_positions, hr_grid_size, hr_box_size, h = read_snapshot(hr_snapshot)
sr_positions, hr_grid_size, hr_box_size, h = read_snapshot(sr_snapshot)

lr_density = cloud_in_cells(lr_positions, lr_grid_size, lr_box_size)
hr_density = cloud_in_cells(hr_positions, hr_grid_size, hr_box_size)
sr_density = cloud_in_cells(sr_positions, hr_grid_size, hr_box_size)


#%% Comological Volume Density Field Comparison.
lr_density_2D = np.sum(lr_density[:, :, :], axis=-1) * 8
hr_density_2D = np.sum(hr_density[:, :, :], axis=-1)
sr_density_2D = np.sum(sr_density[:, :, :], axis=-1)

lr_density_2D = np.log(lr_density_2D)
hr_density_2D = np.log(hr_density_2D)
sr_density_2D = np.log(sr_density_2D)

lr_density_2D = np.rot90(lr_density_2D)
hr_density_2D = np.rot90(hr_density_2D)
sr_density_2D = np.rot90(sr_density_2D)

vmin = min(lr_density_2D.min(), hr_density_2D.min(), sr_density_2D.min())
vmax = max(lr_density_2D.max(), hr_density_2D.max(), sr_density_2D.max())

fig, axes = plt.subplots(1, 3, figsize=(21, 7))

cmap = 'inferno'

im1 = axes[0].imshow(lr_density_2D, cmap=cmap, vmin=vmin, vmax=vmax)
axes[0].set_xticks([])
axes[0].set_yticks([])

im2 = axes[1].imshow(hr_density_2D, cmap=cmap, vmin=vmin, vmax=vmax)
axes[1].set_xticks([])
axes[1].set_yticks([])

im3 = axes[2].imshow(sr_density_2D, cmap=cmap, vmin=vmin, vmax=vmax)
axes[2].set_xticks([])
axes[2].set_yticks([])

box_color = 'black'
rect = patches.Rectangle(
    (0, 63-16), 16, 16, linewidth=4, edgecolor=box_color, facecolor='none'
)
axes[0].add_patch(rect)

rect = patches.Rectangle(
    (0, 127-32), 32, 32, linewidth=4, edgecolor=box_color, facecolor='none'
)
axes[1].add_patch(rect)

rect = patches.Rectangle(
    (0, 127-32), 32, 32, linewidth=4, edgecolor=box_color, facecolor='none'
)
axes[2].add_patch(rect)

plt.tight_layout()
# plt.savefig('cosmo_box_density.png', dpi=300)
plt.show()
plt.close()


#%% Zoom in Density Comparison.
lr_density_2D = np.sum(lr_density[:16, :16, :16], axis=-1) * 8
hr_density_2D = np.sum(hr_density[:32, :32, :32], axis=-1)
sr_density_2D = np.sum(sr_density[:32, :32, :32], axis=-1)

lr_density_2D = np.log(lr_density_2D)
hr_density_2D = np.log(hr_density_2D)
sr_density_2D = np.log(sr_density_2D)

lr_density_2D = np.rot90(lr_density_2D)
hr_density_2D = np.rot90(hr_density_2D)
sr_density_2D = np.rot90(sr_density_2D)

vmin = min(lr_density_2D.min(), hr_density_2D.min(), sr_density_2D.min())
vmax = max(lr_density_2D.max(), hr_density_2D.max(), sr_density_2D.max())

fig, axes = plt.subplots(1, 3, figsize=(21, 7))

cmap = 'inferno'

im1 = axes[0].imshow(lr_density_2D, cmap=cmap, vmin=vmin, vmax=vmax)
axes[0].set_xticks([])
axes[0].set_yticks([])

im2 = axes[1].imshow(hr_density_2D, cmap=cmap, vmin=vmin, vmax=vmax)
axes[1].set_xticks([])
axes[1].set_yticks([])

im3 = axes[2].imshow(sr_density_2D, cmap=cmap, vmin=vmin, vmax=vmax)
axes[2].set_xticks([])
axes[2].set_yticks([])

plt.tight_layout()
# plt.savefig('zoomin_density.png', dpi=300)
plt.show()
plt.close()



#%% Power Spectrum Comparison.
import torch
from dmsr.analysis import power_spectrum

def spectrum(denisty, box_size, grid_size):
    denisty = torch.from_numpy(denisty)
    ks, spectrum, _ = power_spectrum(denisty, box_size, grid_size)
    return ks, spectrum


lr_ks, lr_spectrum = spectrum(lr_density * 8, lr_box_size, lr_grid_size)
hr_ks, hr_spectrum = spectrum(hr_density, hr_box_size, hr_grid_size)
sr_ks, sr_spectrum = spectrum(sr_density, hr_box_size, hr_grid_size)


fig, axes = plt.subplots(
    2, 1, 
    figsize=(8, 6), 
    sharex=True, 
    gridspec_kw={'hspace': 0}
)

axes[0].plot(lr_ks, lr_spectrum, linewidth=4, label='Low Resolution')
axes[0].plot(hr_ks, hr_spectrum, linewidth=4, label='High Resolution')
axes[0].plot(sr_ks, sr_spectrum, linewidth=4, label='Super Resolution')
axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].set_ylabel(r'$\Delta$', fontsize=16)
axes[0].legend()
axes[0].grid()

axes[1].plot(lr_ks, hr_spectrum[:len(lr_spectrum)] / lr_spectrum, linewidth=4)
axes[1].plot(hr_ks, hr_spectrum / hr_spectrum, linewidth=4)
axes[1].plot(hr_ks, hr_spectrum / sr_spectrum, linewidth=4)
axes[1].set_ylim(0.5, 4)
axes[1].set_xlabel('k', fontsize=16)
axes[1].set_ylabel(r'$\Delta / \Delta_{HR}$', fontsize=16)
axes[1].set_yticks([1, 2, 3])
axes[1].grid()
# plt.savefig('power_spectrum.png', dpi=300)
plt.show()
plt.close()


#%% Full Volume Scatter Plots.
lr_xs, lr_ys = lr_positions[:, 0], lr_positions[:, 1]
hr_xs, hr_ys = hr_positions[:, 0], hr_positions[:, 1]
sr_xs, sr_ys = sr_positions[:, 0], sr_positions[:, 1]

fig, axes = plt.subplots(1, 3, figsize=(21, 7))

axes[0].scatter(lr_xs, lr_ys, s=0.02, alpha=0.2)
axes[0].set_xlim(0, lr_box_size)
axes[0].set_ylim(0, lr_box_size)
axes[0].set_xticks([])
axes[0].set_yticks([])

axes[1].scatter(hr_xs, hr_ys, s=0.02, alpha=0.1)
axes[1].set_xlim(0, hr_box_size)
axes[1].set_ylim(0, hr_box_size)
axes[1].set_xticks([])
axes[1].set_yticks([])

axes[2].scatter(sr_xs, sr_ys, s=0.02, alpha=0.1)
axes[2].set_xlim(0, hr_box_size)
axes[2].set_ylim(0, hr_box_size)
axes[2].set_xticks([])
axes[2].set_yticks([])

box_color = 'black'
box_width = 32 * (100 / h) / 128

rect = patches.Rectangle(
    (1, 1), box_width, box_width, linewidth=4, edgecolor=box_color, facecolor='none'
)
axes[0].add_patch(rect)

rect = patches.Rectangle(
    (1, 1), box_width, box_width, linewidth=4, edgecolor=box_color, facecolor='none'
)
axes[1].add_patch(rect)

rect = patches.Rectangle(
    (1, 1), box_width, box_width, linewidth=4, edgecolor=box_color, facecolor='none'
)
axes[2].add_patch(rect)

plt.tight_layout()
# plt.savefig('cosmo_box_scatter.png', dpi=300)
plt.show()
plt.close()


#%% Zoom in Scatter Plots.
upper = 32 * (100 / h) / 128
lower = 0
box_lower = np.asarray([lower, lower, lower])
box_upper = np.asarray([upper, upper, upper])

def get_box_positions(positions, box_lower, box_upper):
    particles_in_box = (box_lower < positions) * (positions < box_upper)
    particles_in_box = np.all(particles_in_box, axis=1)
    particles_in_box = positions[particles_in_box, :]
    return particles_in_box

lr_box_positions = get_box_positions(lr_positions, box_lower, box_upper)
hr_box_positions = get_box_positions(hr_positions, box_lower, box_upper)
sr_box_positions = get_box_positions(sr_positions, box_lower, box_upper)

lr_xs, lr_ys = lr_box_positions[:, 0], lr_box_positions[:, 1]
hr_xs, hr_ys = hr_box_positions[:, 0], hr_box_positions[:, 1]
sr_xs, sr_ys = sr_box_positions[:, 0], sr_box_positions[:, 1]

fig, axes = plt.subplots(1, 3, figsize=(21, 7))

axes[0].scatter(lr_xs, lr_ys, s=0.2, alpha=0.5)
axes[0].set_xlim(lower, upper)
axes[0].set_ylim(lower, upper)
axes[0].set_xticks([])
axes[0].set_yticks([])

axes[1].scatter(hr_xs, hr_ys, s=0.2, alpha=0.5)
axes[1].set_xlim(lower, upper)
axes[1].set_ylim(lower, upper)
axes[1].set_xticks([])
axes[1].set_yticks([])

axes[2].scatter(sr_xs, sr_ys, s=0.2, alpha=0.5)
axes[2].set_xlim(lower, upper)
axes[2].set_ylim(lower, upper)
axes[2].set_xticks([])
axes[2].set_yticks([])

plt.tight_layout()
# plt.savefig('zoomin_scatter.png', dpi=300)
plt.show()
plt.close()