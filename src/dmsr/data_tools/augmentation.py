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
    Randomly selects a composition of axis flips and permutations and applies
    the transformation to both the given LR and HR fields.
    """
    # random_flip = torch.randint(0, 2, (3,), dtype=torch.bool)
    # LR_field = flip_tensor(LR_field, random_flip)
    # HR_field = flip_tensor(HR_field, random_flip)
    
    random_perm = torch.randperm(3)
    LR_field = permute_tensor(LR_field, random_perm)
    HR_field = permute_tensor(HR_field, random_perm)

    return LR_field, HR_field


def random_transformation_single(field):
    """
    Randomly selects a composition of axis flips and permutations and applies
    the transformation to both the given LR and HR fields.
    """
    random_flip = torch.randint(0, 2, (3,)).bool()
    field = flip_tensor(field, random_flip)
    
    random_perm = torch.randperm(3)
    field = permute_tensor(field, random_perm)

    return field


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