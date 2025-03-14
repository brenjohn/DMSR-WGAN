#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 12:49:26 2025

@author: brennan
"""

import torch
import numpy as np


def generate_mock_data(lr_grid_size, hr_grid_size, channels, samples):
    """Create a mock training data set for testing.
    """
    shape = (samples, channels, lr_grid_size, lr_grid_size, lr_grid_size)
    LR_data = torch.rand(*shape)
    shape = (samples, channels, hr_grid_size, hr_grid_size, hr_grid_size)
    HR_data = torch.rand(*shape)
    return LR_data, HR_data