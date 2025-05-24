#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 17 19:47:56 2025

@author: brennan

This script was used to create figure 8 and 9 of Brennan et. al. 2025 "On the 
Use of WGANs for Super Resolution in Dark-Matter Simulations". It produces
plots the present results derived from the minimum spanning tree for hr ans sr
snapshots.
"""

import sys
sys.path.append("..")
sys.path.append("../..")

import h5py as h5
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from swift_tools.minimal_spanning_tree import build_minimal_spanning_tree
from swift_tools.minimal_spanning_tree import compute_branches, preprocess_mst


#%% Read in data
data_dir = './swift_snapshots/test_set/'
hr_snapshot = data_dir + '128/snap_0002.hdf5'
sr_snapshot = data_dir + '064/snap_0002_sr_level_0.hdf5'

file = h5.File(hr_snapshot, 'r')
grid_size = file['ICs_parameters'].attrs['Grid Resolution']
box_size = file['Header'].attrs['BoxSize'][0]
dm_data = file['DMParticles']
hr_positions = np.asarray(dm_data['Coordinates'])

file = h5.File(sr_snapshot, 'r')
grid_size = file['ICs_parameters'].attrs['Grid Resolution']
box_size = file['Header'].attrs['BoxSize'][0]
dm_data = file['DMParticles']
sr_positions = np.asarray(dm_data['Coordinates'])


#%% build MSTs
hr_mst_adj = build_minimal_spanning_tree(hr_positions)
sr_mst_adj = build_minimal_spanning_tree(sr_positions)

hr_mst_adj = hr_mst_adj.tocoo()
sr_mst_adj = sr_mst_adj.tocoo()


#%% Plot edge length distribution before preprocessing
hr_edge_lengths = hr_mst_adj.data
sr_edge_lengths = sr_mst_adj.data
all_lengths = np.concatenate([hr_edge_lengths, sr_edge_lengths])
bins = np.histogram_bin_edges(all_lengths, bins=100)

plt.figure(figsize=(10, 6))
plt.hist(hr_edge_lengths, bins=bins, edgecolor='black', alpha=0.4, color='red')
plt.hist(sr_edge_lengths, bins=bins, edgecolor='black', alpha=0.4, color='blue')
plt.yscale('log')
plt.xlabel("Edge Length (Mpc)")
plt.ylabel("Frequency")
plt.title("Distribution of Edge Lengths in the Minimum Spanning Tree")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
plt.close()


#%% Preprocess the MSTs by pruning k-branches and separating large edges.
hr_mst_graph = nx.from_scipy_sparse_array(hr_mst_adj.tocsr())
sr_mst_graph = nx.from_scipy_sparse_array(sr_mst_adj.tocsr())

k = 25 # Prune branches with k edges.
preprocess_mst(hr_mst_graph, k)
preprocess_mst(sr_mst_graph, k)


#%% Plot branch lengths
hr_branches, hr_branch_lengths, hr_num_segments = compute_branches(hr_mst_graph)
sr_branches, sr_branch_lengths, sr_num_segments = compute_branches(sr_mst_graph)

all_lengths = np.concatenate([hr_branch_lengths, sr_branch_lengths])
bins = np.histogram_bin_edges(all_lengths, bins=100)

plt.figure(figsize=(10, 6))
plt.hist(hr_branch_lengths, bins=bins, edgecolor='black', alpha=0.4, color='red')
plt.hist(sr_branch_lengths, bins=bins, edgecolor='black', alpha=0.4, color='blue')
plt.yscale('log')
plt.xlabel('Branch Length (Mpc)')
plt.ylabel('Frequency')
plt.title(f'Distribution of Branch Lengths in the Minimum Spanning Tree (k={k})')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(f'branch_distribution_k_{k}.png', dpi=210)
plt.show()
plt.close()


#%% Plot distribution of separations of MSTs (see Naidoo 2019 "beyond 
# two-point statistics: using the minimum spanning tree)
hr_branch_ends = [(branch[0], branch[-1]) for branch in hr_branches.values()]
sr_branch_ends = [(branch[0], branch[-1]) for branch in sr_branches.values()]

hr_separations = np.asarray([
    np.linalg.norm(hr_positions[start] - hr_positions[end])
    for start, end in hr_branch_ends
]) / hr_branch_lengths

sr_separations = np.asarray([
    np.linalg.norm(sr_positions[start] - sr_positions[end])
    for start, end in sr_branch_ends
]) / sr_branch_lengths

hr_separations = np.sqrt(1 - hr_separations)
sr_separations = np.sqrt(1 - sr_separations)

all_lengths = np.concatenate([hr_separations, sr_separations])
bins = np.histogram_bin_edges(all_lengths, bins=100)

plt.figure(figsize=(10, 6))
plt.hist(hr_separations, bins=bins, edgecolor='black', alpha=0.4, color='red')
plt.hist(sr_separations, bins=bins, edgecolor='black', alpha=0.4, color='blue')
plt.yscale('log')
plt.xlabel('Separation')
plt.ylabel('Frequency')
plt.title(f'Distribution of Separations in the Minimum Spanning Tree (k={k})')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(f'Separation_distribution_k_{k}.png', dpi=210)
plt.show()
plt.close()


#%% Plot Edge and Branch segments distribution
# Plotting
fig = plt.figure(figsize=(8, 12))
grid = plt.GridSpec(2, 1, hspace=0.18)


# Branch Distribution
ax = fig.add_subplot(grid[1, 0])
all_lengths = np.concatenate([hr_num_segments, sr_num_segments])
bins = np.histogram_bin_edges(all_lengths, bins=50)

ax.hist(
    hr_num_segments, bins=bins, 
    histtype='step', linewidth=2, color='red', label='HR'
)
ax.hist(
    sr_num_segments, bins=bins, 
    histtype='step', linewidth=2, color='blue', label='SR'
)
ax.set_yscale('log')
ax.set_xlabel('Number of links in a branch', fontsize=21)
ax.set_ylabel('Frequency', fontsize=21)
ax.tick_params(labelsize=16)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(fontsize=21)


# Edge Distribution
ax = fig.add_subplot(grid[0, 0])
pruned_hr_mst_adj = nx.to_scipy_sparse_array(hr_mst_graph)
hr_edge_lengths = pruned_hr_mst_adj.data

pruned_sr_mst_adj = nx.to_scipy_sparse_array(sr_mst_graph)
sr_edge_lengths = pruned_sr_mst_adj.data

all_lengths = np.concatenate([hr_edge_lengths, sr_edge_lengths])
bins = np.histogram_bin_edges(all_lengths, bins=25)

ax.hist(
    hr_edge_lengths, bins=bins, 
    histtype='step', linewidth=2, color='red', label='HR'
)
ax.hist(
        sr_edge_lengths, bins=bins, 
        histtype='step', linewidth=2, color='blue', label='SR'
)
ax.set_yscale('log')
ax.set_xlabel('Edge length (Mpc)', fontsize=21)
ax.set_ylabel('Frequency', fontsize=21)
ax.tick_params(labelsize=16)
ax.grid(True, linestyle='--', alpha=0.6)

plt.savefig(f'edge-branch_distribution_k_{k}.png', dpi=210)
plt.show()
plt.close()


#%% plot HR MST
nodes = np.asarray(hr_mst_graph.nodes)
xs = hr_positions[nodes, 0]
ys = hr_positions[nodes, 1]

plt.scatter(xs, ys, s=0.1, alpha=0.1)
plt.xlim(0, box_size)
plt.ylim(0, box_size)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig(f'hr_pruned_positions_k_{k}.png', dpi=210)
plt.show()
plt.close()


#%% plot SR MST
nodes = np.asarray(sr_mst_graph.nodes)
xs = sr_positions[nodes, 0]
ys = sr_positions[nodes, 1]

plt.scatter(xs, ys, s=0.1, alpha=0.1)
plt.xlim(0, box_size)
plt.ylim(0, box_size)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig(f'sr_pruned_positions_k_{k}.png', dpi=210)
plt.show()
plt.close()


#%% Comparison of HR and SR MSTs
fig, (ax_hr, ax_sr) = plt.subplots(1, 2, figsize=(14, 7))

a_nodes = np.asarray([a for a, b in hr_mst_graph.edges])
b_nodes = np.asarray([b for a, b in hr_mst_graph.edges])
xs = np.stack([hr_positions[a_nodes, 0], hr_positions[b_nodes, 0]])
ys = np.stack([hr_positions[a_nodes, 1], hr_positions[b_nodes, 1]])

ax_hr.plot(xs, ys, color='#1f77b4', linewidth=0.5)
ax_hr.set_xlim(0, box_size)
ax_hr.set_ylim(0, box_size)
ax_hr.set_xticks([])
ax_hr.set_yticks([])
ax_hr.text(
    x=120,
    y=130,
    s='HR',
    color='black',
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', lw=2),
    fontsize=35,
    verticalalignment='top',
    horizontalalignment='left'
)


a_nodes = np.asarray([a for a, b in sr_mst_graph.edges])
b_nodes = np.asarray([b for a, b in sr_mst_graph.edges])
xs = np.stack([sr_positions[a_nodes, 0], sr_positions[b_nodes, 0]])
ys = np.stack([sr_positions[a_nodes, 1], sr_positions[b_nodes, 1]])

ax_sr.plot(xs, ys, color='#1f77b4', linewidth=0.5)
ax_sr.set_xlim(0, box_size)
ax_sr.set_ylim(0, box_size)
ax_sr.set_xticks([])
ax_sr.set_yticks([])
ax_sr.text(
    x=120,
    y=130,
    s='SR',
    color='black',
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', lw=2),
    fontsize=35,
    verticalalignment='top',
    horizontalalignment='left'
)

plt.tight_layout()
plt.savefig(f'hr_vs_sr_mst_components_k_{k}.png', dpi=210)
plt.show()
plt.close()


#%% Plot distribution of node degree for both MSTs
hr_degrees = [d for n, d in hr_mst_graph.degree()]
sr_degrees = [d for n, d in sr_mst_graph.degree()]

all_degrees = np.concatenate([hr_degrees, sr_degrees])
bins = np.histogram_bin_edges(all_degrees, bins=5)

plt.figure(figsize=(10, 6))
plt.hist(hr_degrees, bins=bins, edgecolor='black', alpha=0.4, color='red')
plt.hist(sr_degrees, bins=bins, edgecolor='black', alpha=0.4, color='blue')
plt.yscale('log')
plt.xlabel('Node Degree')
plt.ylabel('Frequency')
plt.title(f'Distribution of node degrees in the Minimum Spanning Tree (k={k})')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(f'degree_distribution_k_{k}.png', dpi=210)
plt.show()
plt.close()