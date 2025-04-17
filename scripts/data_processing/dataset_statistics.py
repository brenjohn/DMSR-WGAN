#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 21:31:22 2025

@author: brennan
"""

import os
import sys
sys.path.append("..")
sys.path.append("../..")

import torch
import numpy as np

from dmsr.data_tools.loaders import load_numpy_tensor


def compute_stats(patches):
    """
    Use Welford's algorithm to compute summary statistics of patch arrays in
    the given field directory.
    """
    # Initialize counters
    count = 0
    mean = 0.0
    M2 = 0.0
    
    # for file_path in os.listdir(field_dir):
    for patch_file in patches:
        patch = np.load(patch_file)
        patch = patch.ravel()
        n = patch.size
        new_mean = patch.mean()
        new_M2 = ((patch - new_mean) ** 2).sum()
    
        delta = new_mean - mean
        total = count + n
    
        if total > 0:
            mean += delta * n / total
            M2 += new_M2 + delta**2 * count * n / total
            count = total
    
    variance = M2 / count
    return mean, np.sqrt(variance)


dataset_dir = '../../data/dmsr_style_train/'
fields = [
    'LR_disp_fields', 'HR_disp_fields', 'LR_vel_fields', 'HR_vel_fields'
]

scale_factor_file = dataset_dir + 'scale_factors.npy'
# TODO: Load as numpy tensor with np.load
scale_factors = load_numpy_tensor(scale_factor_file)
unique_scale_factors = torch.unique(scale_factors)

a = unique_scale_factors[0]

#%%
field_dir = dataset_dir + fields[0] + '/'

patches = os.listdir(field_dir)
patches.sort(key = lambda s: (-len(s), s))

scale_stats = {}
for a in unique_scale_factors:
    print('Computing stats for scale factor', a)
    stats = {}
    for field in fields:
        print('Computing stats for', field)
        if not os.path.isdir(dataset_dir + field):
            continue
        
        scale_specific_patches = [
            patches[i] 
            for i, flag in enumerate(scale_factors == a) if flag == True
        ]
        
        scale_specific_patches = [
            dataset_dir + field + '/' + patch 
            for patch in scale_specific_patches
        ]
        
        mean, standard_deviation = compute_stats(scale_specific_patches)
        stats[field + '_std'] = standard_deviation
        stats[field + '_mean'] = mean
        
    scale_stats[a.item()] = stats

print('stats computed:\n', stats)
# np.save(dataset_dir + 'normalisation.npy', stats)

#%%

import matplotlib.pyplot as plt

xs = np.arange(-10, 10, 0.1)

def gauss(xs, mu, std):
    return np.exp(-(xs - mu)**2/(2 * std**2)) / (std * np.sqrt(2*np.pi))

for a in unique_scale_factors:
    std = scale_stats[a.item()]['LR_disp_fields_std']
    mean = scale_stats[a.item()]['LR_disp_fields_mean']
    plt.plot(xs, gauss(xs, mean, std))

plt.xlabel('displacement')
plt.savefig('LR-displacement-dist.png', dpi=210)
plt.show()
plt.close()