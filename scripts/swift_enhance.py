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
import argparse
import numpy as np

from pathlib import Path
from swift_tools.enhancer import DMSREnhancer


def swift_enhance(
        model_dir: Path, 
        data_dir: Path, 
        snapshot_pattern: str, 
        output_suffix: str,
        output_dir: Path,
        seed: int
    ):
    """
    Enhance a Swift snapshots using a DMSR generator model.
    
    Args:
        model_dir (Path): The directory containing the dmsr model files.
        data_dir (Path): The directory containing low-resolution snapshots.
        snapshot_pattern (str): A glob pattern for low-resolution snapshots.
        output_suffix (str): A suffix to add to enhanced snapshot filenames.
        output_dir (Path): Directory to put output files..
    """
    torch.manual_seed(seed)
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create an enhancer object to upscale lr data.
    enhancer = DMSREnhancer(model_dir, device)
    
    # Get paths to low-resolution snapshots.
    lr_snapshots = np.sort(list(data_dir.glob(snapshot_pattern)))
    output_dir.mkdir(exist_ok=True)
    
    for lr_snapshot in lr_snapshots:
        ti = time.time()
        lr_snapshot = Path(lr_snapshot)
        print('Upscaling', lr_snapshot)
        sr_snapshot = output_dir
        sr_snapshot /= f"{lr_snapshot.stem}{output_suffix}{lr_snapshot.suffix}"
        
        # Enhance the low-resolution snapshot
        enhancer.enhance(lr_snapshot, sr_snapshot)
        print(f'Upscaling took {time.time() - ti}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Enhance a Swift snapshot using a PyTorch model."
    )
    
    parser.add_argument(
        '--model_dir', 
        type=Path, 
        default='./dmsr_model/',
        help="Path to the directory containing the generator model."
    )
    parser.add_argument(
        '--data_dir', 
        type=Path, 
        default='./swift_snapshots/',
        help="Path to the directory containing the low-resolution snapshots."
    )
    parser.add_argument(
        '--snapshot_pattern', 
        type=str, 
        default='snap_*.hdf5',
        help="Glob pattern for low-resolution snapshots (e.g., 'snap_*.hdf5')."
    )
    parser.add_argument(
        '--output_suffix', 
        type=str, 
        default='_sr',
        help="Suffix to add to the output filename (e.g., '_sr')."
    )
    parser.add_argument(
        '--output_dir', 
        type=Path, 
        default='./sr_snapshots/',
        help="Directory to put output files."
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=7,
        help="Seed to use for random number generation."
    )
    
    args = parser.parse_args()
    swift_enhance(
        args.model_dir, 
        args.data_dir, 
        args.snapshot_pattern, 
        args.output_suffix,
        args.output_dir,
        args.seed
    )