#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 17:36:56 2024

@author: brennan

This script was used to create figure 3 of Brennan et. al. 2025 "On the Use of
WGANs for Super Resolution in Dark-Matter Simulations".
"""

import numpy as np

from dm_analysis import cloud_in_cells
from swift_tools.data import read_snapshot

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker


#%% Calculating Comological Volume Density Field.
data_dir = './swift_snapshots/meraxes_runs/'
lr_snapshot = data_dir + '064/snap_0099.hdf5'
hr_snapshot = data_dir + '256/snap_0099.hdf5'
sr_snapshot = data_dir + '064/snap_0099_sr.hdf5'

# IDs, positions, velocities, grid_size, box_size, h, mass, a
_, lr_positions, _, lr_grid_size, lr_box_size, h, lr_mass, a = read_snapshot(lr_snapshot)
_, hr_positions, _, hr_grid_size, hr_box_size, h, hr_mass, a = read_snapshot(hr_snapshot)
_, sr_positions, _, hr_grid_size, hr_box_size, h, sr_mass, a = read_snapshot(sr_snapshot)

lr_density = cloud_in_cells(lr_positions, lr_grid_size, lr_box_size)
hr_density = cloud_in_cells(hr_positions, hr_grid_size, hr_box_size)
sr_density = cloud_in_cells(sr_positions, hr_grid_size, hr_box_size)


#%% Comological Volume Density Field Comparison.
lr_density_2D = np.sum(lr_density[:, :, :], axis=-1) * lr_mass * 1e10 / lr_box_size
hr_density_2D = np.sum(hr_density[:, :, :], axis=-1) * hr_mass * 1e10 / hr_box_size
sr_density_2D = np.sum(sr_density[:, :, :], axis=-1) * sr_mass * 1e10 / hr_box_size

lr_density_2D = np.log10(lr_density_2D)
hr_density_2D = np.log10(hr_density_2D)
sr_density_2D = np.log10(sr_density_2D)

lr_density_2D = np.rot90(lr_density_2D)
hr_density_2D = np.rot90(hr_density_2D)
sr_density_2D = np.rot90(sr_density_2D)

vmin = min(lr_density_2D.min(), hr_density_2D.min(), sr_density_2D.min())
vmax = max(lr_density_2D.max(), hr_density_2D.max(), sr_density_2D.max())

fig, axes = plt.subplots(
    1, 3,
    figsize=(24, 7),
    gridspec_kw={
        'width_ratios': [1, 1, 1],
    }
)

# Manually set uniform subplot params for consistent heights (override tight_layout adjustments)
fig.subplots_adjust(left=0.03, right=0.92, top=0.97, bottom=0.03, wspace=0.05)

cmap = 'inferno'

im1 = axes[0].imshow(lr_density_2D, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
axes[0].set_xticks([])
axes[0].set_yticks([])

im2 = axes[1].imshow(hr_density_2D, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
axes[1].set_xticks([])
axes[1].set_yticks([])

im3 = axes[2].imshow(sr_density_2D, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
axes[2].set_xticks([])
axes[2].set_yticks([])

# Create colorbar attached to the last image axis for matching height
divider = make_axes_locatable(axes[2])
cax = divider.append_axes("right", size="5%", pad=0.28)
cbar = fig.colorbar(im3, cax=cax)
cbar.set_label(r'$\log_{10}(\rho)$', fontsize=28)
cbar.ax.tick_params(labelsize=21)
cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.1f}'))
# cbar.ax.set_yticks([10, 10.5, 11, 11.5])

box_color = 'white'
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

# No tight_layout here to avoid compressing the third subplot
plt.savefig('cosmo_box_density.png', dpi=300, bbox_inches='tight')  # bbox_inches='tight' for final padding in saved file
# plt.show()
plt.close()


# # Place the color bar in the 4th subplot
# fig.subplots_adjust(right=0.8)
# cbar_ax = fig.add_axes([0.8, 0.2, 0.05, 0.60])
# cbar = fig.colorbar(im3, cax=cbar_ax, shrink=1.0, aspect=20)
# cbar.set_label(r'$\log_{10}(\rho)$', fontsize=18)

# #%% Zoom in Density Comparison.
# # Note, lr particles are 8 times more massive than hr particles.
# lr_density_2D = np.sum(lr_density[48:, :16, :16], axis=-1) * 8
# hr_density_2D = np.sum(hr_density[96:, :32, :32], axis=-1)
# sr_density_2D = np.sum(sr_density[96:, :32, :32], axis=-1)

# lr_density_2D = np.log(lr_density_2D)
# hr_density_2D = np.log(hr_density_2D)
# sr_density_2D = np.log(sr_density_2D)

# lr_density_2D = np.rot90(lr_density_2D)
# hr_density_2D = np.rot90(hr_density_2D)
# sr_density_2D = np.rot90(sr_density_2D)

# vmin = min(lr_density_2D.min(), hr_density_2D.min(), sr_density_2D.min())
# vmax = max(lr_density_2D.max(), hr_density_2D.max(), sr_density_2D.max())

# fig, axes = plt.subplots(1, 3, figsize=(21, 7))

# cmap = 'inferno'

# im1 = axes[0].imshow(lr_density_2D, cmap=cmap, vmin=vmin, vmax=vmax)
# axes[0].set_xticks([])
# axes[0].set_yticks([])

# im2 = axes[1].imshow(hr_density_2D, cmap=cmap, vmin=vmin, vmax=vmax)
# axes[1].set_xticks([])
# axes[1].set_yticks([])

# im3 = axes[2].imshow(sr_density_2D, cmap=cmap, vmin=vmin, vmax=vmax)
# axes[2].set_xticks([])
# axes[2].set_yticks([])

# plt.tight_layout()
# plt.savefig('zoomin_density.png', dpi=300)
# plt.show()
# plt.close()
