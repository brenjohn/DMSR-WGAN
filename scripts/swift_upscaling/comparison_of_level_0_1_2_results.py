#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 15:56:00 2025

@author: brennan

This script was used to create figure 8 of Brennan et. al. 2025 "On the Use of
WGANs for Super Resolution in Dark-Matter Simulations".
"""

import sys
sys.path.append("..")
sys.path.append("../..")

import numpy as np
import matplotlib.pyplot as plt

from swift_tools.density import cloud_in_cells
from swift_tools.data import read_snapshot


#%% Calculating Density Field.
data_dir = './swift_snapshots/'
l0_snapshot = data_dir + '064/snap_0002_sr_level_0.hdf5'
l1_snapshot = data_dir + '064/snap_0002_sr_level_1.hdf5'
l2_snapshot = data_dir + '064/snap_0002_sr_level_2.hdf5'

l0_positions, l0_grid_size, l0_box_size, h, _ = read_snapshot(l0_snapshot)
l1_positions, l1_grid_size, l1_box_size, h, _ = read_snapshot(l1_snapshot)
l2_positions, l2_grid_size, l2_box_size, h, _ = read_snapshot(l2_snapshot)

l0_density = cloud_in_cells(l0_positions, l0_grid_size, l0_box_size)
l1_density = cloud_in_cells(l1_positions, l1_grid_size, l1_box_size)
l2_density = cloud_in_cells(l2_positions, l2_grid_size, l2_box_size)


#%% Zoom in Density Comparison.
l0_density_2D = np.sum(l0_density[:32, :32, :32], axis=-1)
l1_density_2D = np.sum(l1_density[:32, :32, :32], axis=-1)
l2_density_2D = np.sum(l2_density[:32, :32, :32], axis=-1)

l0_density_2D = np.log(l0_density_2D)
l1_density_2D = np.log(l1_density_2D)
l2_density_2D = np.log(l2_density_2D)

l0_density_2D = np.rot90(l0_density_2D)
l1_density_2D = np.rot90(l1_density_2D)
l2_density_2D = np.rot90(l2_density_2D)

vmin = min(l0_density_2D.min(), l1_density_2D.min(), l2_density_2D.min())
vmax = max(l0_density_2D.max(), l1_density_2D.max(), l2_density_2D.max())

fig, axes = plt.subplots(1, 3, figsize=(21, 7))

cmap = 'inferno'

axes[0].imshow(l0_density_2D, cmap=cmap, vmin=vmin, vmax=vmax)
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[0].text(
    x=0.5,
    y=2,
    s='0 layers',
    color='white',
    fontsize=35,
    verticalalignment='top',
    horizontalalignment='left'
)

axes[1].imshow(l1_density_2D, cmap=cmap, vmin=vmin, vmax=vmax)
axes[1].set_xticks([])
axes[1].set_yticks([])
axes[1].text(
    x=0.5,
    y=2,
    s='1 layers',
    color='white',
    fontsize=35,
    verticalalignment='top',
    horizontalalignment='left'
)

axes[2].imshow(l2_density_2D, cmap=cmap, vmin=vmin, vmax=vmax)
axes[2].set_xticks([])
axes[2].set_yticks([])
axes[2].text(
    x=0.5,
    y=2,
    s='2 layers',
    color='white',
    fontsize=35,
    verticalalignment='top',
    horizontalalignment='left'
)

plt.tight_layout()
plt.savefig('level_comparison_zoomin_density.png', dpi=300)
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

l0_box_positions = get_box_positions(l0_positions, box_lower, box_upper)
l1_box_positions = get_box_positions(l1_positions, box_lower, box_upper)
l2_box_positions = get_box_positions(l2_positions, box_lower, box_upper)

l0_xs, l0_ys = l0_box_positions[:, 0], l0_box_positions[:, 1]
l1_xs, l1_ys = l1_box_positions[:, 0], l1_box_positions[:, 1]
l2_xs, l2_ys = l2_box_positions[:, 0], l2_box_positions[:, 1]

fig, axes = plt.subplots(1, 3, figsize=(21, 7))

axes[0].scatter(l0_xs, l0_ys, s=0.2, alpha=0.5)
axes[0].set_xlim(lower, upper)
axes[0].set_ylim(lower, upper)
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[0].text(
    x=1,
    y=33,
    s='0 layers',
    color='black',
    fontsize=35,
    verticalalignment='top',
    horizontalalignment='left'
)

axes[1].scatter(l1_xs, l1_ys, s=0.2, alpha=0.5)
axes[1].set_xlim(lower, upper)
axes[1].set_ylim(lower, upper)
axes[1].set_xticks([])
axes[1].set_yticks([])
axes[1].text(
    x=1,
    y=33,
    s='1 layers',
    color='black',
    fontsize=35,
    verticalalignment='top',
    horizontalalignment='left'
)

axes[2].scatter(l2_xs, l2_ys, s=0.2, alpha=0.5)
axes[2].set_xlim(lower, upper)
axes[2].set_ylim(lower, upper)
axes[2].set_xticks([])
axes[2].set_yticks([])
axes[2].text(
    x=1,
    y=33,
    s='2 layers',
    color='black',
    fontsize=35,
    verticalalignment='top',
    horizontalalignment='left'
)

plt.tight_layout()
plt.savefig('level_comparison_zoomin_scatter.png', dpi=300)
plt.show()
plt.close()