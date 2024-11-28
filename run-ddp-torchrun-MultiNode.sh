#!/bin/bash
#SBATCH --job-name=ddp-dist      # create a short name for your job
#SBATCH --nodes=2               # node count
#SBATCH --gres=gpu:2             # number of gpus per node
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-task=72        # cpu-cores per task (>1 if multi-threaded tasks)
######SBATCH --mem=900G               # total memory per node (4 GB per cpu-core is default)
#SBATCH --mem=100G               # total memory per node (4 GB per cpu-core is default)
#SBATCH --time=00:20:00          # total run time limit (HH:MM:SS)


#SBATCH -p gpu
#SBATCH -A OD-230112
#export CUDA_VISIBLE_DEVICES=2,3

module load apptainer slurm
module load pytorch/2.3.1-py312-cu122-mpi
# zoom zoom 
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_MIN_CHANNELS=32


export RDZV_HOST=$(hostname)
export RDZV_PORT=29400                   


# DEBUG  if you want to print all debug options 
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSY=ALL
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

USERNAME=dia021


# Assumes you've built container prior using it, see README
# You can build your own container from NGC 23.10 or 23.11 or earlier 
#srun apptainer exec --nv /scratch3/$USERNAME/containers/trchprosthesis_requirements_24.02-py3.sif torchrun \
srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=2\
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$RDZV_HOST:$RDZV_PORT" \
    mnist_ddp.py --epochs=100 

