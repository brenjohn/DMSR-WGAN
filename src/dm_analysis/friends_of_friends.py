#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 12:50:35 2025

@author: john

This file has implementations of the friends-of-friends algorithm. 
"""

import numpy as np

from scipy.spatial import cKDTree
from collections import defaultdict


#=============================== Union Find ==================================#

class UnionFind:
    def __init__(self, size):
        self.parent = np.arange(size)

    def find(self, x):
        # Path compression.
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        # Find representatives of y and x.
        x_root = self.find(x)
        y_root = self.find(y)
        
        # Merge y into x
        if x_root != y_root:
            self.parent[y_root] = x_root


def friends_of_friends(positions, box_size, linking_length):
    """
    Return a list of halos in the given particle position data. 
    
    Uses a union find algorithm to identify halos. Periodic boundary conditions
    are assumed.
    """
    n_particles = len(positions)
    tree = cKDTree(positions, boxsize=box_size)
    uf = UnionFind(n_particles)

    for i in range(n_particles):
        neighbors = tree.query_ball_point(positions[i], linking_length)
        for j in neighbors:
            if i < j:  # Only union each pair once
                uf.union(i, j)

    # Build halos from groups
    halos = defaultdict(list)
    for i in range(n_particles):
        leader = uf.find(i)
        halos[leader].append(i)

    return list(halos.values())


#============================= BFS flood fill ================================#

def friends_of_friends_bfs(positions, box_size, linking_length):
    """
    Return a list of halos in the given particle position data.
    
    Uses a breadth first search algorithm to identify halos. Periodic boundary 
    conditions are assumed.
    """
    tree = cKDTree(positions, boxsize=box_size)
    
    n_particles = len(positions)
    visited = np.zeros(n_particles, dtype=bool)
    halos = []
    
    for i in range(n_particles):
        if not visited[i]:
            # Start a new halo
            halo = []
            stack = [i]
            
            while stack:
                idx = stack.pop()
                halo.append(idx)
                visited[idx] = True
                neighbours = tree.query_ball_point(
                    positions[idx], linking_length
                )
                
                neighbours = [n for n in neighbours if not visited[n]]
                stack.extend(neighbours)
                for neighbour in neighbours:
                    visited[neighbour] = True
            
            halos.append(halo)
    
    return halos