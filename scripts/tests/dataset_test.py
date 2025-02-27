#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 19:09:23 2024

@author: brennan
"""

import sys
sys.path.append("..")
sys.path.append("../..")

import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from swift_tools.data import load_numpy_dataset
from dmsr.dmsr_gan import DMSRDataset
from dmsr.field_operations.conversion import displacements_to_positions


#%%


def plot_positions(positions):
    positions = torch.transpose(positions, 1, -1)
    positions = positions.reshape((-1, 3))
    xs = positions[:, 0]
    ys = positions[:, 1]
    
    plt.scatter(xs, ys, alpha=0.2, s=0.1)
    plt.show()
    plt.close()



data_directory = '../../data/dmsr_training_velocity_x64/'
data = load_numpy_dataset(data_directory)
LR_data, HR_data, box_size, LR_grid_size, HR_grid_size = data
LR_data, HR_data = LR_data[:7, ...], HR_data[:7, ...]

dataset = DMSRDataset(LR_data, HR_data, augment=True)


#%%

# DataLoader to iterate over the dataset
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Example: Iterate through the dataset
for lr_batch, hr_batch in dataloader:
    print(lr_batch.shape)
    print(hr_batch.shape)
    
    lr_positions = displacements_to_positions(lr_batch, 20*box_size/16) - (2 * box_size/16)
    hr_positions = displacements_to_positions(hr_batch, box_size)
    plot_positions(lr_positions)
    plot_positions(hr_positions)
    plt.show()
    plt.close()


#%%
disp, bin_edges, _ = plt.hist(LR_data[:, 0:3, ...].reshape(-1), bins=50)
plt.show()
plt.close()

vel, bin_edges, _ = plt.hist(LR_data[:, 3:6, ...].reshape(-1)/50, bins=50)
plt.show()
plt.close()