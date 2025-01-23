#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 14:31:25 2024

@author: john

This script uses a specified dmsr generator to enhance the dark-matter data in
a low-resolution swift snapshot.
"""

import sys
sys.path.append("..")
sys.path.append("../..")

import time
import torch

from swift_tools.enhance import enhance

device = "cpu"
print(f"Using device: {device}")
ti = time.time()

# Load the generator model
dmsr_model_dir = './trained_model_levels/level_2/current_model/'
generator = torch.load(dmsr_model_dir + 'generator.pth').to(device)

# Specify paths to low-resolution snapshot and where to save enhanced snapshot. 
data_dir = './swift_snapshots/'
lr_snapshot = data_dir + '064/snap_0002.hdf5'
sr_snapshot = lr_snapshot.replace('.hdf5', '_sr_level_2_tmp.hdf5')

# Enhance the low-resolution snapshot
enhance(lr_snapshot, sr_snapshot, generator, device)
print(f'Upscaling took {time.time() - ti}')