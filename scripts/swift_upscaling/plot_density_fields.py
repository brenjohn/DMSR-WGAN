#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 13:39:11 2025

@author: brennan
"""

import h5py as h5
import numpy as np
from pathlib import Path
from swift_tools.data import read_metadata

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Directory containing snapshots and density fields.
data_dir = Path('./swift_snapshots/meraxes_runs/256/')

# Glob patterns for density fields and snapshots
density_pattern = 'snap_*_density.npy'
snapshot_pattern = 'snap_*.hdf5'

density_files = np.sort(list(data_dir.glob(density_pattern)))
snapshot_files = np.sort(list(data_dir.glob(snapshot_pattern)))

cmap = 'inferno'

vmin_manual = 12.139329057587343
vmax_manual = 15.301921156070636

for density_file, snapshot_file in zip(density_files, snapshot_files):
    grid_size, box_size, h, mass, a = read_metadata(snapshot_file)
    
    density = np.load(density_file)
    density_2D = np.sum(density[:, :, :], axis=-1) * mass * 1e10 * box_size
    density_2D = np.log10(density_2D)
    density_2D = np.rot90(density_2D)
    
    vmin = vmin_manual if vmin_manual is not None else density_2D.min()
    vmax = vmax_manual if vmax_manual is not None else density_2D.min()
    
    plt.figure(figsize=(7, 7))
    plt.imshow(density_2D, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    plt.xticks([])
    plt.yticks([])
    
    # z = int(1/a - 1)
    # box_color = 'white'
    # plt.text(
    #     x=1,
    #     y=4,
    #     s=f'HR\nz = {z}',
    #     color='white',
    #     fontsize=35,
    #     fontweight='bold',
    #     verticalalignment='top',
    #     horizontalalignment='left'
    # )
    
    plot_file = data_dir / (str(density_file.stem) + '.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()