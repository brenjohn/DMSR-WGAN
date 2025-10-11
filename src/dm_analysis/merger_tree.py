#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  7 15:25:35 2025

@author: brennan
"""

import pickle
from collections import defaultdict


def build_merger_tree(halo_catalogues):
    """
    Build a merger tree from a list of halo catalogues. Each catalogue contains 
    a list of halos from a particular snapshot. Ancestor-descendant
    relationships between halos from consecutive snapshots are identified using
    a particle matching method and the corresponding halos are linked together
    by updating their ancestor and descendant attributes.
    
    halo_catalogues is a list of halo catalougues from different snapshots.
    Each halo catalogue is a list of halo objects whose halo ids are tuples
    of the form (snap_ind, halo_ind) where snap_ind is the index of the halo
    catalogue in `halo_catalogues` and halo_ind is the index of the halo in
    its halo catalogue.
    """

    for t in range(len(halo_catalogues) - 1):
        curr_catalogue = halo_catalogues[t]
        next_catalogue = halo_catalogues[t + 1]

        # Build particle-to-halo lookup for the next catalogue
        particle_to_halo_next = {}
        for halo in next_catalogue:
            for pid in halo.particle_ids:
                particle_to_halo_next[pid] = halo.halo_id

        # Build matches from current halos to next halos
        for halo in curr_catalogue:
            counts = defaultdict(int)

            # Count how many particles match each halo in next catalogue
            for pid in halo.particle_ids:
                if pid in particle_to_halo_next:
                    matched_halo_id = particle_to_halo_next[pid]
                    counts[matched_halo_id] += 1

            if counts:
                # Find the best match by highest number of shared particles
                best_match = max(counts.items(), key=lambda x: x[1])[0]
                snap_ind, halo_ind = best_match
                descendant = halo_catalogues[snap_ind][halo_ind]
                link_halos(halo, descendant)
                
        del particle_to_halo_next


def link_halos(ancestor, descendant):
    descendant.set_ancestor(ancestor.halo_id)
    ancestor.set_descendant(descendant.halo_id)
    
    
def save_halo_catalogues(halo_catalogues, filename):
    """Saves a nested list of Halo objects to a file."""
    with open(filename, 'wb') as f:
        pickle.dump(halo_catalogues, f)

    
def load_halo_catalogues(filename):
    """Loads a nested list of Halo objects from a pickled file."""
    with open(filename, 'rb') as f:
        halo_catalogues = pickle.load(f)
    return halo_catalogues