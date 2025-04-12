#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 23:31:43 2025

@author: john
"""

import sys
sys.path.append("..")
sys.path.append("../..")

import numpy as np
import matplotlib.pyplot as plt

from swift_tools.data import read_snapshot
from swift_tools.friends_of_friends import friends_of_friends
from swift_tools.friends_of_friends import Halo, compute_shape


def moment_of_inertia(positions):
    """Computes the moment of inertia tensor from the given particle positions.
    """
    I = np.zeros((3, 3))
    identity = np.eye(3)
    
    for r in positions:
        I += np.dot(r,r) * identity - np.outer(r, r)
    
    return I


#%% Read in snapshot data.
data_dir = './swift_snapshots/'
lr_snapshot = data_dir + '064/snap_0002.hdf5'
hr_snapshot = data_dir + '128/snap_0002.hdf5'
sr_snapshot = data_dir + '064/snap_0002_sr_level_1.hdf5'

hr_positions, _, hr_box_size, h, hr_mass = read_snapshot(hr_snapshot)
sr_positions, _, sr_box_size, h, sr_mass = read_snapshot(sr_snapshot)


#%% Compute the mass of halos.
def halo_catalogue(positions, box_size, mass):
    """Returns a list of masses for all friends-of-friends halos.
    """
    b = 0.2 # Sets the linking length fraction.
    N = 100 # Only consider halos with at least this many particles.
    volume = box_size**3
    mean_interparticle_separation = (volume / len(positions))**(1/3)
    linking_length = b * mean_interparticle_separation
    halos = friends_of_friends(positions, box_size, linking_length)
    return [
        Halo.from_halo_dict(halo_dict, mass) 
        for halo_dict in halos if len(halo_dict) > N
    ]



print('Getting hr halos')
hr_halos = halo_catalogue(hr_positions, hr_box_size, hr_mass * 1e10)

print('Getting sr halos')
sr_halos = halo_catalogue(sr_positions, sr_box_size, sr_mass * 1e10)


#%%
def get_shapes(halos, particle_mass):
    masses = []
    bas = []
    cas = []
    for i, halo in enumerate(halos):
        print('Halo', i)
        ratios = compute_shape(halo, 35)
        if ratios is not None:
            ba, ca = ratios
            bas.append(ba)
            cas.append(ca)
            masses.append(halo.mass)
    return masses, bas, cas

hr_masses, hr_bas, hr_cas = get_shapes(hr_halos, hr_mass)
sr_masses, sr_bas, sr_cas = get_shapes(sr_halos, sr_mass)


#%%
plt.scatter(hr_masses, hr_bas, label='HR')
plt.scatter(sr_masses, sr_bas, label='SR')
plt.xscale('log')
plt.xlabel('Mass', fontsize=14)
plt.ylabel('b / a', fontsize=14)
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig('ba.png', dpi=210)
plt.show()
plt.close()


#%%
plt.scatter(hr_masses, hr_cas, label='HR')
plt.scatter(sr_masses, sr_cas, label='SR', alpha=1)
plt.xscale('log')
plt.xlabel('Mass', fontsize=14)
plt.ylabel('c / a', fontsize=14)
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig('ca_level_1.png', dpi=210)
plt.show()
plt.close()


#%%
bins = np.logspace(12, 16, 5)
hr_bin_indices = np.digitize(hr_masses, bins)
sr_bin_indices = np.digitize(sr_masses, bins)

hr_means = []
sr_means = []
hr_bin_centers = []
sr_bin_centers = []

for i in range(1, len(bins)):
    in_bin = (hr_bin_indices == i)
    if np.any(in_bin):
        median_shape = np.median(np.asarray(hr_cas)[in_bin])
        hr_means.append(median_shape)
        hr_bin_centers.append((bins[i] + bins[i - 1]) / 2)
        
    in_bin = (sr_bin_indices == i)
    if np.any(in_bin):
        median_shape = np.median(np.asarray(sr_cas)[in_bin])
        sr_means.append(median_shape)
        sr_bin_centers.append((bins[i] + bins[i - 1]) / 2)
        
plt.plot(hr_bin_centers, hr_means, lw=2, label='Median HR shape')
plt.plot(sr_bin_centers, sr_means, lw=2, label='Median SR shape')
plt.xscale('log')
plt.ylim(0, 1)
plt.xlabel('Mass', fontsize=14)
plt.ylabel(r'$c/a$', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig('median_shape_level_1.png', dpi=210)
plt.show()
plt.close()


#%%
plt.scatter(hr_bas, hr_cas, label='HR', s=np.log(hr_masses))
plt.scatter(sr_bas, sr_cas, label='SR', s=np.log(sr_masses))
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
# plt.xscale('log')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('b / a')
plt.ylabel('c / a')
plt.legend()
plt.tight_layout()
plt.savefig('ba_vs_ca_level_1.png', dpi=210)
plt.show()
plt.close()


#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# Helper: Custom colormap with white at low density
def shifted_cmap(base_cmap, ncolors=256, start_white_frac=0.14):
    base = plt.get_cmap(base_cmap)
    colors = base(np.linspace(0, 1, ncolors))
    colors[:int(ncolors * start_white_frac)] = [1, 1, 1, 1]
    return ListedColormap(colors)

# Prepare Data
data1 = np.stack([np.asarray(hr_bas), np.asarray(hr_cas)])
data2 = np.stack([np.asarray(sr_bas), np.asarray(sr_cas)])

kde1 = gaussian_kde(data1)
kde2 = gaussian_kde(data2)

# Grid setup
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
grid_coords = np.vstack([X.ravel(), Y.ravel()])

Z1 = kde1(grid_coords).reshape(X.shape)
Z2 = kde2(grid_coords).reshape(X.shape)

# Custom Colormaps
reds_with_white = shifted_cmap("Reds")
blues_with_white = shifted_cmap("Blues")

# Plotting
plt.figure(figsize=(8, 8))

# Filled contour plots with transparency
contour1 = plt.contourf(X, Y, Z1, levels=3, cmap=reds_with_white, alpha=0.5)
contour2 = plt.contourf(X, Y, Z2, levels=3, cmap=blues_with_white, alpha=0.5)

# Optional diagonal line for reference
plt.plot([0, 1], [0, 1], color='black', linestyle='--')

# Contour outlines (label if desired)
plt.contour(X, Y, Z1, levels=3, colors='red', linewidths=0.5)
plt.contour(X, Y, Z2, levels=3, colors='blue', linewidths=0.5)

# Labels
plt.xlabel(r'$b / a$', fontsize=21)
plt.ylabel(r'$c / a$', fontsize=21)

# Legend with matching colors
hr_patch = mpatches.Patch(color='red', alpha=0.6, label='HR')
sr_patch = mpatches.Patch(color='blue', alpha=0.6, label='SR')
plt.legend(handles=[hr_patch, sr_patch], fontsize=21)

# Show plot
plt.tight_layout()
plt.savefig('shape_distribution_level_1.png', dpi=210)
plt.show()

#%%
# compute_shape(sr_halos[384], 35)
sr_halos[384].positions

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

n = 100

xs = sr_halos[384].positions[:, 0]
ys = sr_halos[384].positions[:, 1]
zs = sr_halos[384].positions[:, 2]
ax.scatter(xs, ys, zs)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()