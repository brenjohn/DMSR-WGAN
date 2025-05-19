#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 19:52:47 2024

@author: brennan
"""

import os
import sys
sys.path.append("..")
sys.path.append("../..")

import numpy as np
import matplotlib.pyplot as plt

from swift_tools.fields import get_positions
from dmsr.field_operations.augmentation import random_transformation


def patch_location(n, patches_per_snapshot, patches_per_dim):
    n = n % patches_per_snapshot
    z = n % patches_per_dim
    y = (n % (patches_per_dim**2)) // patches_per_dim
    x = n // (patches_per_dim**2)
    return x, y, z


dataset_dir = '../../data/dmsr_style_train/'
fields = [
    'LR_disp_fields', 'HR_disp_fields', 'LR_vel_fields', 'HR_vel_fields'
]

scale_factor_file = dataset_dir + 'scale_factors.npy'
scale_factors = np.load(scale_factor_file)

metadata = np.load(dataset_dir + 'metadata.npy')
box_size        = metadata[0]
LR_patch_length = metadata[1]
HR_patch_length = metadata[2]
LR_patch_size   = metadata[3]
HR_patch_size   = metadata[4]
LR_inner_size   = metadata[5]
padding         = metadata[6]
LR_mass         = metadata[7]
HR_mass         = metadata[8]

#%%
field_dir = dataset_dir + fields[0] + '/'

patches = os.listdir(field_dir)
patches.sort(key = lambda s: (len(s), s))

LR_disp_patches = [
    dataset_dir + 'LR_disp_fields/' + patch for patch in patches
]

HR_disp_patches = [
    dataset_dir + 'HR_disp_fields/' + patch for patch in patches
]

#%%
for n in range(len(LR_disp_patches)-64, len(LR_disp_patches)):
    LR_patch = LR_disp_patches[n]
    HR_patch = HR_disp_patches[n]
    a = scale_factors[n][0]
    
    LR_disp = np.load(LR_patch)[:, 2:18, 2:18, 2:18]
    HR_disp = np.load(HR_patch)
        
    LR_positions = get_positions(LR_disp, a * box_size / 4, 16)
    HR_positions = get_positions(HR_disp, a * box_size / 4, 32)
    
    print('Patch location', patch_location(n, 64, 4))
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    axs[0].scatter(HR_positions[0, ...], HR_positions[1, ...], s=1, alpha=0.1)
    # plt.show()
    # plt.close()
    
    axs[1].scatter(LR_positions[0, ...], LR_positions[1, ...], s=5, alpha=0.9)
    plt.title(f'Patch location {patch_location(n, 64, 4)}')
    plt.show()
    plt.close()
    
    
#%%
n = 49002
LR_patch = LR_disp_patches[n]
HR_patch = HR_disp_patches[n]
a = scale_factors[n][0]

LR_disp = np.load(LR_patch)[:, 2:18, 2:18, 2:18]
HR_disp = np.load(HR_patch)
    
LR_positions = get_positions(LR_disp, a * box_size / 4, 16)
HR_positions = get_positions(HR_disp, a * box_size / 4, 32)

plt.scatter(HR_positions[0, ...], HR_positions[1, ...], s=1, alpha=0.9)
plt.scatter(LR_positions[0, ...], LR_positions[1, ...], s=2, alpha=0.9)
plt.show()
plt.close()


# LR_file = '../../data/dmsr_training/LR_fields.npy'
# LR_fields = np.load(LR_file)

# HR_file = '../../data/dmsr_training/HR_fields.npy'
# HR_fields = np.load(HR_file)

# meta_file = '../../data/dmsr_training/metadata.npy'
# box_size, LR_grid_size, HR_grid_size = np.load(meta_file)
# LR_grid_size = int(LR_grid_size)
# HR_grid_size = int(HR_grid_size)


# #%%
# plt.style.use('dark_background')

# n = 70
# LR_field = LR_fields[n, ...]
# HR_field = HR_fields[n, ...]

# LR_xs = get_positions(LR_field, box_size, LR_grid_size, False)
# HR_xs = get_positions(HR_field, box_size, HR_grid_size, False)

# figure, axes = plt.subplots(1, 2, figsize=(10, 5))
# axes[0].scatter(LR_xs[0, :], LR_xs[1, :], alpha=0.3, s=0.1, color='white')
# axes[1].scatter(HR_xs[0, :], HR_xs[1, :], alpha=0.2, s=0.1, color='white')
# axes[0].set_yticks([])
# axes[0].set_xticks([])
# axes[1].set_yticks([])
# axes[1].set_xticks([])

# plt.subplots_adjust(wspace=0, hspace=0)
# plt.show()


# #%%
# dataset = tf.data.Dataset.from_tensor_slices(
#     (LR_fields[:6, ...], HR_fields[:6, ...])
# )
# dataset = dataset.map(random_transformation)
# dataset = dataset.shuffle(len(dataset))


# #%%
# num = 0
# for LR, HR in dataset:

#     LR = LR.numpy()
#     HR = HR.numpy()
    
#     LR_xs = get_positions(LR, box_size, LR_grid_size, False)
#     HR_xs = get_positions(HR, box_size, HR_grid_size, False)
    
#     figure, axes = plt.subplots(1, 2, figsize=(10, 5))
#     axes[0].scatter(LR_xs[0, :], LR_xs[1, :], alpha=0.3, s=0.1, color='white')
#     axes[1].scatter(HR_xs[0, :], HR_xs[1, :], alpha=0.2, s=0.1, color='white')
#     axes[0].set_yticks([])
#     axes[0].set_xticks([])
#     axes[1].set_yticks([])
#     axes[1].set_xticks([])
#     plt.subplots_adjust(wspace=0, hspace=0)
#     plt.savefig(f'sample{num}.png', dpi=200)
#     plt.show()
#     plt.close()
#     num += 1