#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:38:30 2024

@author: john
"""

import sys
sys.path.append("..")
sys.path.append("../..")

import h5py as h5
import numpy as np

import matplotlib.pyplot as plt


#%%
data_dir = './swift_snapshots/'
snapshot = data_dir + '064/snap_0002_sr.hdf5'

file = h5.File(snapshot, 'r')

dm_data = file['DMParticles']

positions = np.asarray(dm_data['Coordinates'])
xs = positions[:, 0]
ys = positions[:, 1]


#%%
plt.scatter(xs, ys, alpha=0.05, s=0.01)
plt.show()
plt.close()