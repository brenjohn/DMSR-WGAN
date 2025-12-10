#!/bin/bash
#SBATCH --job-name=dmsr_job
#SBATCH --chdir=./
#SBATCH --output=dmsr_%j.out
#SBATCH --error=dmsr_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1  
#SBATCH --gpus-per-node=4   
#SBATCH --cpus-per-task=20

module purge
module load mkl/2024.0 nvidia-hpc-sdk/23.11-cuda11.8 openblas/0.3.27-gcc cudnn/9.0.0-cuda11 tensorrt/10.0.0-cuda11 impi/2021.11 hdf5/1.14.1-2-gcc gcc/11.4.0 python/3.11.5-gcc nccl/2.19.4
module load pytorch/2.4.0

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

srun torchrun \
--nnodes 2 \
--nproc_per_node 4 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:29500 \
../scripts/dmsr_train.py --parameter_file=./training_parameters.json
