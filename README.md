# DMSR-WGAN: Dark-Matter Super-Resolution WGAN

This repository contains code and resources aimed at enhancing the particle resolution of dark matter-only simulations using a Wasserstein Generative Adversarial Network (WGAN). 
This work has applications in astrophysics and cosmology, particularly for improving simulation fidelity in large-scale structure studies.

---
## Table of Contents
1. [Project Overview](#project-overview)
2. [Dependencies](#dependencies)
3. [Results](#reproducing-results)
<!-- 4. [Usage](#usage) -->

---
## Project Overview

The **DMSR-WGAN** uses adversarial training to super-resolve dark matter simulations. It generates high-resolution particle distributions from low-resolution inputs. This approach enables:
- Enhanced particle resolution.
- Improved substructure identification.
- Reduced computational cost compared to high-resolution simulations.

For more details, refer to: 
[On the Use of WGANs for Super Resolution in Dark-Matter Simulations](https://arxiv.org/abs/2501.13056 ).

---
## Dependencies

Note, this project used the following dependencies:
- **Python**: Version 3.12.5
- **HDF5**: Version 3.11.0
- **PyTorch**: Version 2.4.0
- **Matplotlib**: Version 3.9.2
- **NumPy**: Version 1.26.4
- **SciPy**: Version 1.13.1

---
## Reproducing Results

Results were produced for [On the Use of WGANs for Super Resolution in Dark-Matter Simulations](https://arxiv.org/abs/2501.13056 ) using the workflow outlined below. Please refer to the paper for further details on architecture and parameters used.

1. **Generate Initial Conditions for Dark-Matter-Only Simulations**
   - Initial conditions for dark-matter-only simulations where generated using [MUSIC](https://bitbucket.org/ohahn/music/src/master/).
   - Parameter files for generating 17 pairs of initial conditions, with each pair consiting of a low-resolution and high-resolution realisation, can be found in `data/dmsr_runs/`.
   - Each simulation pair has a dedicated directory `runAB` containing directories for the low-resolution and high-resolution simulations.
   - Initial conditions were generated within each simulation directory using the `ics.conf` MUSIC parameter file.

2. **Run Simulations with SWIFT**
   - The dark-matter-only simulations were run in each simulation directory using [SWIFT code](https://swift.dur.ac.uk/) and the `dmsr_run.yml` parameter file.
   - Snapshots from the simulation were stored as HDF5 files in each simulation directory for processing in the next step.

3. **Prepare Datasets**
   - Training and validation datasets were then created from all simulation snapshots created in the last step.
   - This was done using the `scripts/data_processing/dataset_preperation.py` script. This reads particle position data from the SWIFT snapshots and creates a set of low- and high-resolution displacement fields.
   - The displacement fields are written to `data/dmsr_X/HR_fields.npy` and `data/dmsr_X/LR_fields.npy` where `X` is either `training` or `validation`. Some metadata is also written to `data/dmsr_X/metadata.npy`.

4. **Training a DMSR-WGAN**
   - The `scripts/training/GAN_training.py` script was then used to configure and train a DMSR-WGAN.
   - This script saves trained models in `scripts/training/output_dir/checkpoints/`. During training, the most up-to-date model is saved in the `current_model/` sub-directory.

5. **Enhancing a SWIFT Snapshot**
   - To Enhance a snapshot with a trained DMSR-WGAN, the WGAN was copied to `scripts/swift_upscaling/dmsr_model`. The simulation directory containing the snapshot to be enhanced was also copied to `scripts/swift_upscaling/swift_snapshots`.
   - The `scripts/swift_upscaling/swift_enhance.py` script was then used to create an enhanced version of a snapshot saved in `swift_snapshots` and save it to the same simulation directory.
   - The enhanced snapshot was then analysed using the analysis scripts in `scripts/swift_upscaling/`.
