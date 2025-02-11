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
import numpy as np

from os.path import exists
from swift_tools.enhance import enhance

device = "cpu"
print(f"Using device: {device}")
ti = time.time()

# Load the generator model.
dmsr_model_dir = './dmsr_model/'
generator = torch.load(dmsr_model_dir + 'generator.pth').to(device)

# Load any scaling parameters if they exist.
scale_path = dmsr_model_dir + "normalisation.npy"
scale_params = None
if exists(scale_path):
    scale_params = np.load(scale_path, allow_pickle=True).item()
    # TODO: scale parameters shouldn't be stored as tensors
    scale_params = {k : v.item() for k, v in scale_params.items()}

# Specify paths to low-resolution snapshot and where to save enhanced snapshot. 
data_dir = './swift_snapshots/'
lr_snapshot = data_dir + '064/snap_0002.hdf5'
sr_snapshot = lr_snapshot.replace('.hdf5', '_sr.hdf5')

# Enhance the low-resolution snapshot
enhance(lr_snapshot, sr_snapshot, generator, scale_params, device)
print(f'Upscaling took {time.time() - ti}')