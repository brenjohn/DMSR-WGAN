#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 17 18:16:34 2025

@author: brennan
"""

import numpy as np
import networkx as nx

from scipy.spatial import KDTree
from scipy.sparse import coo_array, csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree


def build_minimal_spanning_tree(positions, k = 10):
    """Returns the adjacency matrix for the minimal spanning tree for the
    given particle position data. The minimal spanning tree is computed from
    the k-nearest-neighbour graph of the particles.
    """
    num_particles = positions.shape[0]
    
    print('Constructing KDTree')
    tree = KDTree(positions)

    print('Constructing kNN graph')
    distances, indices = tree.query(positions, k=k+1)
    rows = np.repeat(np.arange(num_particles), k).astype(np.intc)
    cols = indices[:, 1:].flatten().astype(np.intc)
    vals = distances[:, 1:].flatten()
    kNN_adjacency = coo_array((vals, (rows, cols)))
    
    print('Constructing minimal spanning tree')
    return minimum_spanning_tree(kNN_adjacency.tocsr())


#=============================================================================#
#                      functions for networkx trees                           #
#=============================================================================#


def prune(tree, branches, max_length):
    """Removes any of the given branches from the given tree that have length
    less than or equal to max_length.
    
    Returns the number of edges that were removed in the prcoess.
    """
    edges_removed = 0
    
    for branch in branches.values():
        if len(branch) > max_length:
            continue
        
        for node in branch[:-1]:
            if tree.has_node(node):
                tree.remove_node(node)
            else:
                break
            
        edges_removed += len(branch) - 1
            
    return edges_removed
      
        
def separate(tree, tau = 3):
    """Removes edges from the graph that are longer than `tau` times the 
    average edge weight.
    
    Returns the number of edges that were removed in the prcoess.
    """
    weights = nx.get_edge_attributes(tree, 'weight')
    avg_weight = sum(weights.values()) / len(weights)
    
    edges_to_remove = [
        edge for edge, weight in weights.items() 
        if weight > tau * avg_weight
    ]
    
    tree.remove_edges_from(edges_to_remove)
    return len(edges_to_remove)


def remove_null_graphs(tree):
    for component in list(nx.connected_components(tree)):
        if len(component) == 1:
            tree.remove_nodes_from(component)
            
            
def compute_branches(tree):
    """Identifies and tracks branches in a tree starting from leaf nodes, 
    where a branch is a chain of degree-2 nodes leading to a node 
    with degree not equal to 2 (e.g., a junction or another leaf).
    
    Returns:
        branches (dict): dict of lists of nodes in a branch. Indexed by leaves.
        branch_lengths (list): Total weight of each branch.
        branch_num_segments (list): Number of segments (edges) in each branch. 
    """
    branches = {}
    branch_lengths = []
    branch_num_segments = []
    
    leaf_nodes = [
        node for node, degree in dict(tree.degree()).items() 
        if degree == 1
    ]
    
    for leaf in leaf_nodes:
        
        curr_node = leaf
        next_node = list(tree.neighbors(curr_node))[0]
        branch = [curr_node, next_node]
        branch_length = tree[curr_node][next_node]['weight'] 
        num_segments = 0
        
        while tree.degree(next_node) == 2:
            prev_node = curr_node
            curr_node = next_node
            node_a, node_b = list(tree.neighbors(next_node))
            next_node = node_a if node_b == prev_node else node_b
            
            branch.append(next_node)
            branch_length += tree[curr_node][next_node]['weight']
            num_segments += 1
        
        branches[leaf] = branch
        branch_lengths.append(branch_length)
        branch_num_segments.append(num_segments)
    
    return branches, branch_lengths, branch_num_segments


def preprocess_mst(mst_graph, k=15, tau=3):
    """Prunes and separates the given minimal spanning tree as discussed in
    Barrow, Bhavsar, Sonoda 1985 (Minimal spanning trees, fílaments and galaxy 
    clustering).
    
    This is achieved by iteratively removing edges that are `tau` times longer
    than the average and pruning all branches with `k` segements/edges or less
    until to no more edges can be removed.
    """
    removed_edges = 1
    pruned_edges = 1
    iteration = 0
    
    while removed_edges > 0 or pruned_edges > 0:
        removed_edges = separate(mst_graph, tau)
        branches, _, branch_num_segments = compute_branches(mst_graph)
        
        pruned_edges = prune(mst_graph, branches, k)
        message = f'Iteration {iteration}: Removed {removed_edges} edges'
        message += f' and pruned {pruned_edges} edges'
        print(message)
        remove_null_graphs(mst_graph)
        iteration += 1