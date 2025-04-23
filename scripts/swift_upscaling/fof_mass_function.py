#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:35:02 2025

@author: brennan

This script was used to create figure 5 of Brennan et. al. 2025 "On the Use of
WGANs for Super Resolution in Dark-Matter Simulations".
"""

import sys
sys.path.append("..")
sys.path.append("../..")

import matplotlib.pyplot as plt

from numpy import log10, histogram
from swift_tools.data import read_snapshot
from swift_tools.friends_of_friends import friends_of_friends


#%% Read in snapshot data.
data_dir = './swift_snapshots/'
lr_snapshot = data_dir + '064/snap_0002.hdf5'
hr_snapshot = data_dir + '128/snap_0002.hdf5'
sr_snapshot = data_dir + '064/snap_0002_sr_level_0.hdf5'

lr_positions, _, lr_box_size, h, lr_mass = read_snapshot(lr_snapshot)
hr_positions, _, hr_box_size, h, hr_mass = read_snapshot(hr_snapshot)
sr_positions, _, sr_box_size, h, sr_mass = read_snapshot(sr_snapshot)


#%% Compute the mass of halos.
def halo_masses(positions, box_size, mass):
    """Returns a list of masses for all friends-of-friends halos.
    """
    b = 0.2 # Sets the linking length fraction.
    N = 100 # Only consider halos with at least this many particles.
    volume = box_size**3
    mean_interparticle_separation = (volume / len(positions))**(1/3)
    linking_length = b * mean_interparticle_separation
    halos = friends_of_friends(positions, box_size, linking_length)
    return [mass * 1e10 * len(halo) for halo in halos if len(halo) > N]


print('Getting lr halos')
lr_halos = halo_masses(lr_positions, lr_box_size, lr_mass)
print('Getting hr halos')
hr_halos = halo_masses(hr_positions, hr_box_size, hr_mass)
print('Getting sr halos')
sr_halos = halo_masses(sr_positions, sr_box_size, sr_mass)


#%% Compute and plot the number density.
volume = (hr_box_size)**3 # volume in Mpc^3

sr_halo_mass, sr_bin_edges = histogram(log10(sr_halos), bins=9)
hr_halo_mass, hr_bin_edges = histogram(log10(hr_halos), bins=9)
lr_halo_mass, lr_bin_edges = histogram(log10(lr_halos), bins=hr_bin_edges)

fig = plt.Figure(figsize=(7, 7))

bins = (lr_bin_edges[:-1] + lr_bin_edges[1:])/2
plt.plot(bins, log10(lr_halo_mass/volume), 
         linewidth=4, label='Low Resolution', zorder=3)

bins = (hr_bin_edges[:-1] + hr_bin_edges[1:])/2
plt.plot(bins, log10(hr_halo_mass/volume), 
         linewidth=4, label='High Resolution', zorder=1)

bins = (sr_bin_edges[:-1] + sr_bin_edges[1:])/2
plt.plot(bins, log10(sr_halo_mass/volume), 
         linewidth=4, label='Super Resolution', zorder=2)

plt.xticks([13, 13.5, 14, 14.5])
plt.yticks([-6.0, -5.5, -5, -4.5, -4])
plt.xlim((12.95, 14.5))
plt.ylim((-6.05, -3.8))

plt.tick_params(axis='both', which='major', labelsize=14)
plt.xlabel(r'Halo Mass$ \quad [\log_{10} M_\odot]$', fontsize=16)
plt.ylabel(r'Number Density$ \quad [\log_{10} $Mpc$^{-3}]$', fontsize=16)
plt.legend(fontsize=14)
plt.grid()
plt.tight_layout()
plt.savefig('halo_mass_function.png', dpi=210)
plt.show()
plt.close()