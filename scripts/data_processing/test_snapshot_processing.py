#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 17:02:27 2024

@author: brennan

This script creates some plots from the DM particles position data in a swift
snapshot. This script is used for data inspection and testing functions for
creating displacement fields.
"""

import os
import sys
sys.path.append("..")
sys.path.append("../..")

import h5py as h5
import numpy as np

from dmsr.swift_processing import get_displacement_field, get_positions


#%% Load snapshot data.
snapshot_path = '../../data/dmsr_runs/run01/128/snap_0002.hdf5'

snapshot    = h5.File(snapshot_path, 'r')
coordinates = np.asarray(snapshot['DMParticles']['Coordinates']).transpose()
particleIDs = np.asarray(snapshot['DMParticles']['ParticleIDs'])
grid_size   = snapshot['ICs_parameters'].attrs['Grid Resolution']
box_size    = snapshot['Header'].attrs['BoxSize'][0]

displacement_field = get_displacement_field(
    coordinates, 
    particleIDs, 
    box_size, 
    grid_size
)

positions = get_positions(displacement_field, box_size, grid_size)


#%% Plot the distribution of values in the original coordinates, the
# displacement field and the reconstructed positions of particles. The first
# two distributions should be identical and the third plot should look like a
# Gaussian.
import matplotlib.pyplot as plt

plt.figure()
plt.hist(coordinates.reshape(-1), bins = 100, density = True)
plt.title('Coordinate Distribution')
plt.show()
plt.close()

plt.figure()
plt.hist(positions.reshape(-1), bins = 100, density = True)
plt.title('Reconstructed Coordinate Distribution')
plt.show()
plt.close()

plt.figure()
plt.hist(displacement_field.reshape(-1), bins = 100, density = True)
plt.title('Particle Displacements')
plt.show()
plt.close()


#%% The following scatter plots of particle positions should be identical.
plt.figure()
plt.scatter(coordinates[0, :], coordinates[1, :], alpha=0.07, s=0.025)
plt.title('Particle Positions')
plt.show()
plt.close()

plt.figure()
plt.scatter(positions[0, :], positions[1, :], alpha=0.07, s=0.025)
plt.title('Reconstructed Particle Positions')
plt.show()
plt.close()