#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:35:02 2025

@author: brennan
"""

import sys
sys.path.append("..")
sys.path.append("../..")

import h5py as h5
import numpy as np
from scipy.spatial import cKDTree

def create_pbc_ghosts(positions, box_size, linking_length):
    """
    Create ghost particles for periodic boundary conditions.
    """
    offsets = [-1, 0, 1]  # Represents shifts of -1, 0, +1 box size
    shifts = np.array([
        (dx, dy, dz) for dx in offsets for dy in offsets for dz in offsets 
        if (dx, dy, dz) != (0, 0, 0)
    ])
    ghost_positions = []
    for shift in shifts:
        ghost_positions.append(positions + shift * box_size)
    
    return np.vstack([positions] + ghost_positions)


def friend_of_friends(positions, box_size, linking_length):
    extended_positions = create_pbc_ghosts(positions, box_size, linking_length)
    tree = cKDTree(extended_positions)
    
    n_particles = len(positions)
    visited = np.zeros(len(extended_positions), dtype=bool)
    halos = []
    
    for i in range(n_particles):
        if not visited[i]:
            # Start a new halo
            halo = []
            stack = [i]
            
            while stack:
                idx = stack.pop()
                if not visited[idx]:
                    visited[idx] = True
                    halo.append(idx)
                    # Find neighbors
                    neighbors = tree.query_ball_point(
                        extended_positions[idx], linking_length
                    )
                    stack.extend(neighbors)
            halos.append(halo)
    
    return halos


def read_snapshot(snapshot):
    file = h5.File(snapshot, 'r')
    
    h = file['Cosmology'].attrs['h'][0]
    
    grid_size = file['ICs_parameters'].attrs['Grid Resolution']
    box_size = file['Header'].attrs['BoxSize'][0]
    
    dm_data = file['DMParticles']
    positions = np.asarray(dm_data['Coordinates'])
    mass = np.asarray(dm_data['Masses'])[0]
    
    file.close()
    return positions, grid_size, box_size, h, mass



#%%
data_dir = './swift_snapshots/'
lr_snapshot = data_dir + '064/snap_0002.hdf5'
hr_snapshot = data_dir + '128/snap_0002.hdf5'
sr_snapshot = data_dir + '064/snap_0002_sr.hdf5'

lr_positions, lr_grid_size, lr_box_size, h, lr_mass = read_snapshot(lr_snapshot)
hr_positions, hr_grid_size, hr_box_size, h, hr_mass = read_snapshot(hr_snapshot)
sr_positions, sr_grid_size, sr_box_size, h, sr_mass = read_snapshot(sr_snapshot)


#%%
def halos(positions, box_size):
    # Example usage
    b = 0.2 # Adjust linking length fraction as needed
    volume = box_size**3
    
    mean_interparticle_separation = (volume / len(positions))**(1/3)
    
    linking_length = b * mean_interparticle_separation
     
    halos = friend_of_friends(positions, box_size, linking_length)
    return halos


print('Getting lr halos')
lr_halos = halos(lr_positions, lr_box_size)
print('Getting hr halos')
hr_halos = halos(hr_positions, hr_box_size)
print('Getting sr halos')
sr_halos = halos(sr_positions, sr_box_size)

#%%
lr_halo_size = [lr_mass * 1e10 * len(halo) for halo in lr_halos if len(halo) > 100]
hr_halo_size = [hr_mass * 1e10 * len(halo) for halo in hr_halos if len(halo) > 100]
sr_halo_size = [sr_mass * 1e10 * len(halo) for halo in sr_halos if len(halo) > 100]

#%%
import matplotlib.pyplot as plt

volume = (hr_box_size * h)**3 # volume in Mpc^3

sr_halo_mass, sr_bin_edges = np.histogram(np.log10(sr_halo_size), bins=9)
hr_halo_mass, hr_bin_edges = np.histogram(np.log10(hr_halo_size), bins=9)
lr_halo_mass, lr_bin_edges = np.histogram(np.log10(lr_halo_size), bins=hr_bin_edges)

fig = plt.Figure(figsize=(7, 7))

bins = (lr_bin_edges[:-1] + lr_bin_edges[1:])/2
plt.plot(bins, np.log10(lr_halo_mass/volume), 
         linewidth=4, label='Low Resolution', zorder=3)

bins = (hr_bin_edges[:-1] + hr_bin_edges[1:])/2
plt.plot(bins, np.log10(hr_halo_mass/volume), 
         linewidth=4, label='High Resolution', zorder=1)

bins = (sr_bin_edges[:-1] + sr_bin_edges[1:])/2
plt.plot(bins, np.log10(sr_halo_mass/volume), 
         linewidth=4, label='Super Resolution', zorder=2)

plt.xticks([13, 13.5, 14, 14.5])
plt.yticks([-5.5, -5, -4.5, -4, -3.5])
plt.xlim((12.9, 14.5))
plt.ylim((-5.6, -3.3))

plt.tick_params(axis='both', which='major', labelsize=14)

plt.xlabel(r'Halo Mass$ \quad [\log_{10} M_\odot]$', fontsize=16)
plt.ylabel(r'Number Density$ \quad [\log_{10} $Mpc$^{-3}]$', fontsize=16)
plt.legend(fontsize=14)
plt.grid()
plt.tight_layout()
plt.savefig('halo_mass_function.png', dpi=210)
plt.show()
plt.close()