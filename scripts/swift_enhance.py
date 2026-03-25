#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 14:31:25 2024

@author: john

This script uses a specified dmsr generator to enhance the dark-matter data in
a low-resolution swift snapshot.
"""

import torch
import time
import tomllib
import argparse
import numpy as np

from pathlib import Path
from swift_tools.enhancer import DMSREnhancer


def swift_enhance(
        model_dir:        Path, 
        data_dir:         Path, 
        snapshot_pattern: str, 
        output_suffix:    str,
        output_dir:       Path,
        seed:             int,
        use_compression:  bool
    ):
    """
    Enhance a Swift snapshots using a DMSR generator model.
    
    Args:
        model_dir        : The directory containing the dmsr model files.
        data_dir         : The directory containing low-resolution snapshots.
        snapshot_pattern : A glob pattern for low-resolution snapshots.
        output_suffix    : A suffix to add to enhanced snapshot filenames.
        output_dir       : Directory to put output files..
    """
    torch.manual_seed(seed)
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create an enhancer object to upscale lr data.
    enhancer = DMSREnhancer(model_dir, device)
    
    # Get paths to low-resolution snapshots.
    lr_snapshots = np.sort(list(data_dir.glob(snapshot_pattern)))
    output_dir.mkdir(exist_ok=True)
    if not lr_snapshots.size: print("Warning: No files found!")
    
    for lr_snapshot in lr_snapshots:
        ti = time.time()
        print('Upscaling', lr_snapshot)
        sr_snapshot = output_dir
        sr_snapshot /= f"{lr_snapshot.stem}{output_suffix}{lr_snapshot.suffix}"
        
        # Enhance the low-resolution snapshot
        enhancer.enhance(lr_snapshot, sr_snapshot, use_compression)
        print(f'Upscaling took {time.time() - ti}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Enhance a Swift snapshot using a PyTorch model."
    )
    parser.add_argument(
        '--parameter_file', 
        type=Path,
        help="Path to the parameter toml file."
    )
    args = parser.parse_args()
    
    with open(args.parameter_file, 'rb') as file:
        params = tomllib.load(file)
    
    for name in ['model_dir', 'data_dir', 'output_dir']:
        params[name] = Path(params[name])
    
    swift_enhance(**params)