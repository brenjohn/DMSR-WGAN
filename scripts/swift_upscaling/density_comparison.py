#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 17:36:56 2024

@author: brennan

This script was used to create figure 3 of Brennan et. al. 2025 "On the Use of
WGANs for Super Resolution in Dark-Matter Simulations".
"""

import sys
sys.path.append("..")
sys.path.append("../..")

import numpy as np

from swift_tools.density import cloud_in_cells
from swift_tools.data import read_snapshot

import matplotlib.pyplot as plt
import matplotlib.patches as patches


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

box_color = 'white'
rect = patches.Rectangle(
    (0, 63-16), 16, 16, linewidth=4, edgecolor=box_color, facecolor='none'
)
axes[0].add_patch(rect)
axes[0].text(
    x=0.5,
    y=2,
    s='LR',
    color='white',
    fontsize=35,
    fontweight='bold',
    verticalalignment='top',
    horizontalalignment='left'
)

rect = patches.Rectangle(
    (0, 127-32), 32, 32, linewidth=4, edgecolor=box_color, facecolor='none'
)
axes[1].add_patch(rect)
axes[1].text(
    x=1,
    y=4,
    s='HR',
    color='white',
    fontsize=35,
    fontweight='bold',
    verticalalignment='top',
    horizontalalignment='left'
)

rect = patches.Rectangle(
    (0, 127-32), 32, 32, linewidth=4, edgecolor=box_color, facecolor='none'
)
axes[2].add_patch(rect)
axes[2].text(
    x=1,
    y=4,
    s='SR',
    color='white',
    fontsize=35,
    fontweight='bold',
    verticalalignment='top',
    horizontalalignment='left'
)

plt.tight_layout()
plt.savefig('cosmo_box_density.png', dpi=300)
plt.show()
plt.close()


#%% Zoom in Density Comparison.
# Note, lr particles are 8 times more massive than hr particles.
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
plt.savefig('zoomin_density.png', dpi=300)
plt.show()
plt.close()