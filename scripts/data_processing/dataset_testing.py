#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 19:52:47 2024

@author: brennan
"""

import sys
sys.path.append("..")
sys.path.append("../..")

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from dmsr.swift_processing import get_positions
from dmsr.operations.augmentation import random_transformation


LR_file = '../../data/dmsr_training/LR_fields.npy'
LR_fields = np.load(LR_file)

HR_file = '../../data/dmsr_training/HR_fields.npy'
HR_fields = np.load(HR_file)

meta_file = '../../data/dmsr_training/metadata.npy'
box_size, LR_grid_size, HR_grid_size = np.load(meta_file)
LR_grid_size = int(LR_grid_size)
HR_grid_size = int(HR_grid_size)


#%%
plt.style.use('dark_background')

n = 70
LR_field = LR_fields[n, ...]
HR_field = HR_fields[n, ...]

LR_xs = get_positions(LR_field, box_size, LR_grid_size, False)
HR_xs = get_positions(HR_field, box_size, HR_grid_size, False)

figure, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].scatter(LR_xs[0, :], LR_xs[1, :], alpha=0.3, s=0.1, color='white')
axes[1].scatter(HR_xs[0, :], HR_xs[1, :], alpha=0.2, s=0.1, color='white')
axes[0].set_yticks([])
axes[0].set_xticks([])
axes[1].set_yticks([])
axes[1].set_xticks([])

plt.subplots_adjust(wspace=0, hspace=0)
plt.show()


#%%
dataset = tf.data.Dataset.from_tensor_slices(
    (LR_fields[:6, ...], HR_fields[:6, ...])
)
dataset = dataset.map(random_transformation)
dataset = dataset.shuffle(len(dataset))


#%%
num = 0
for LR, HR in dataset:

    LR = LR.numpy()
    HR = HR.numpy()
    
    LR_xs = get_positions(LR, box_size, LR_grid_size, False)
    HR_xs = get_positions(HR, box_size, HR_grid_size, False)
    
    figure, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].scatter(LR_xs[0, :], LR_xs[1, :], alpha=0.3, s=0.1, color='white')
    axes[1].scatter(HR_xs[0, :], HR_xs[1, :], alpha=0.2, s=0.1, color='white')
    axes[0].set_yticks([])
    axes[0].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_xticks([])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(f'sample{num}.png', dpi=200)
    plt.show()
    plt.close()
    num += 1