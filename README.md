# DMSR-WGAN: Dark-Matter Super-Resolution WGAN

This repository contains code and resources aimed at enhancing the particle resolution of dark matter-only simulations using a Wasserstein Generative Adversarial Network (WGAN). 
This work has applications in astrophysics and cosmology, particularly for improving simulation fidelity in large-scale structure studies.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Reproducing Results](#reproducing-results)
5. [License](#license)

---

## Project Overview

The **DMSR-WGAN** uses adversarial training to super-resolve dark matter simulations. It generates high-resolution particle distributions from low-resolution inputs. This approach enables:
- Enhanced particle resolution.
- Improved substructure identification.
- Reduced computational cost compared to high-resolution simulations.

For more details, refer to: 
[On the Use of WGANs for Super Resolution in Dark-Matter Simulations](https://arxiv.org/abs/2501.13056 ).

---

## Installation

The required dependencies for this project are listed in the `pyproject.toml` file. You can install them using pip:

```bash
pip install numpy scipy matplotlib torch h5py networkx
```

---

## Usage

Parameters for creating datasets and training a model should be set in parameter toml files. Example parameter files can be found in the `example` directory.

All the following commands should be run from the project root directory.

To create training and validation datasets, it is assumed a collection or LR-HR swift snapshots exist in a directory whose path is specified in a toml file. The following command will create a training dataset and place it in a directory specified in the used toml file. The same command can be used to create a validation set.
```
python scripts/dataset_preparation.py --config_file=example/dataset_config_train.toml --num_procs=4
```

The following commands will compute summary statistics for the training set (used for rescaling data for the WGAN) and compute power spectra for patches (used by the spectrum monitor for validation).
```
python scripts/dataset_statistics.py --dataset_dir=data/dmsr_train/ --include_velocities=True
python scripts/dataset_spectra.py --dataset_dir=data/dmsr_valid/
```

A training job can be launched with the following command:
```
python scripts/dmsr_train.py --parameter_file=example/training_parameters.toml
```

Once a model is trained, it can be used to upscale a swift snapshot with a command like the following:
```
python scripts/swift_enhance.py --model_dir=example/sr_model/current_model/ --data_dir=data/swift_sims/run1/064/ --snapshot_pattern=snap_000[0-8].hdf5 --output_suffix=_sr --output_dir=example/sr_snapshots --seed=7
```

---

## Reproducing Results
For scripts and steps to reproduce results from [On the Use of WGANs for Super Resolution in Dark-Matter Simulations](https://arxiv.org/abs/2501.13056 ), 
please switch to the `on-wgan-super-resolution` branch.

---

## License
This project is licensed under the MIT License - see the [LICENCE.txt](LICENCE.txt) file for details.
