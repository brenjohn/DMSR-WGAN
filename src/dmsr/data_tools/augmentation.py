#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 19:04:03 2024

@author: brennan

This file defines functions for randomly augmenting data with spatial rotations
and permutations.
"""

import torch


def random_transformation(LR_field, HR_field):
    """
    Randomly selects a composition of axis flips and permutations.
    """
    random_flip = torch.randint(0, 2, (3,), dtype=torch.bool)
    random_perm = torch.randperm(3)
    return random_flip, random_perm


def flip_tensor(tensor, flip):
    """
    Flips the given tensor along the specified axes.
    """
    flip_dims = torch.where(flip)[0] + 1
    flip_dims = flip_dims.tolist()
    tensor = tensor.flip(flip_dims)
    tensor[flip, ...] = -1 * tensor[flip, ...]
    return tensor.detach()


def permute_tensor(tensor, permutation):
    """
    Permutes the dimensions of the given tensor.
    """
    dims_permutation = [0] + (permutation + 1).tolist()
    tensor = tensor.permute(dims_permutation)
    tensor = tensor[permutation, ...]
    return tensor.detach()