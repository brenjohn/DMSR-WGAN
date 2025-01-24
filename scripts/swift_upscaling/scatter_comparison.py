#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 17:36:56 2024

@author: brennan

This script was used to create figure 6 of Brennan et. al. 2025 "On the Use of
WGANs for Super Resolution in Dark-Matter Simulations".
"""

import sys
sys.path.append("..")
sys.path.append("../..")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from swift_tools.data import read_snapshot


#%% Calculating Comological Volume Density Field.
data_dir = './swift_snapshots/'
lr_snapshot = data_dir + '064/snap_0002.hdf5'
hr_snapshot = data_dir + '128/snap_0002.hdf5'
sr_snapshot = data_dir + '064/snap_0002_sr.hdf5'

lr_positions, lr_grid_size, lr_box_size, h, _ = read_snapshot(lr_snapshot)
hr_positions, hr_grid_size, hr_box_size, h, _ = read_snapshot(hr_snapshot)
sr_positions, hr_grid_size, hr_box_size, h, _ = read_snapshot(sr_snapshot)


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
    (1, 1), box_width, box_width, linewidth=4,
    edgecolor=box_color, facecolor='none'
)
axes[0].add_patch(rect)
axes[0].text(
    x=2,
    y=135,
    s='LR',
    color='black',
    fontsize=35,
    fontweight='bold',
    verticalalignment='top',
    horizontalalignment='left'
)

rect = patches.Rectangle(
    (1, 1), box_width, box_width, linewidth=4,
    edgecolor=box_color, facecolor='none'
)
axes[1].add_patch(rect)
axes[1].text(
    x=2,
    y=135,
    s='HR',
    color='black',
    fontsize=35,
    fontweight='bold',
    verticalalignment='top',
    horizontalalignment='left'
)

rect = patches.Rectangle(
    (1, 1), box_width, box_width, linewidth=4,
    edgecolor=box_color, facecolor='none'
)
axes[2].add_patch(rect)
axes[2].text(
    x=2,
    y=135,
    s='SR',
    color='black',
    fontsize=35,
    fontweight='bold',
    verticalalignment='top',
    horizontalalignment='left'
)

plt.tight_layout()
plt.savefig('cosmo_box_scatter.png', dpi=210)
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
plt.savefig('zoomin_scatter.png', dpi=300)
plt.show()
plt.close()