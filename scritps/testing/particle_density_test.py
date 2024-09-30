#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 15:26:18 2024

@author: john
"""

import sys
sys.path.append("..")
sys.path.append("../..")

import time
import torch
import matplotlib.pyplot as plt

from dmsr.swift_processing import load_numpy_dataset
from dmsr.field_operations.conversion import displacements_to_positions
from dmsr.field_operations.conversion import cic_density_field



#%%
data_directory = '../../data/dmsr_training/'
data = load_numpy_dataset(data_directory)
LR_data, HR_data, box_size, LR_grid_size, HR_grid_size = data
HR_data = HR_data[:1, ...]


#%% Creating a scatter plot of particle positions.
def plot_positions(positions):
    positions = torch.transpose(positions, 1, -1)
    positions = positions.reshape((-1, 3))
    xs = positions[:, 0]
    ys = positions[:, 1]
    
    plt.Figure(figsize=(8, 8))
    plt.scatter(xs, ys, alpha=0.2, s=0.1)
    plt.tight_layout()
    plt.show()
    plt.close()

HR_positions = displacements_to_positions(HR_data, box_size)
plot_positions(HR_positions)



#%% Create heat map of cic density field
ti = time.time()
density = cic_density_field(HR_data, box_size)
print(f'cic time: {time.time() - ti}')
density = density[0, 0, :, :, :]
density = torch.sum(density, axis=-1)
density = torch.transpose(density, 0, 1)
density = torch.flip(density, dims=[0])

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
plt.imshow(density)#, vmin=vmin, vmax=vmax)
ax.set_title('Density Plot')
plt.tight_layout()
plt.show()
# fig.savefig('cic_density_before.png', dpi=300)
plt.close()

