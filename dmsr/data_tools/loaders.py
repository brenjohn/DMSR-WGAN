#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:56:32 2025

@author: brennan
"""

import torch
import numpy as np

from os.path import exists


def load_numpy_tensor(filename):
    """Returns data contained in the given numpy file.
    """
    return torch.from_numpy(np.load(filename)).float()


def load_normalisation_parameters(param_file):
    """Reads the standard deviations from the given .npy file used to noramlise
    dmsr training data.
    """
    lr_pos_std = hr_pos_std = lr_vel_std = hr_vel_std = 1
    
    if exists(param_file):
        scale_params = np.load(param_file, allow_pickle=True).item()
        lr_pos_std = scale_params.get('lr_position_std', 1)
        hr_pos_std = scale_params.get('hr_position_std', 1)
        lr_vel_std = scale_params.get('lr_velocity_std', 1)
        hr_vel_std = scale_params.get('hr_velocity_std', 1)
    
    return lr_pos_std, hr_pos_std, lr_vel_std, hr_vel_std