#!/bin/bash
#SBATCH --job-name=ddp-dist      # create a short name for your job
#SBATCH --nodes=1               # node count
#SBATCH --gres=gpu:4             # number of gpus per node
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-task=72        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=30G               # total memory per node (4 GB per cpu-core is default)
#SBATCH --time=02:00:00          # total run time limit (HH:MM:SS)

module load apptainer # 


# zoom zoom 
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_MIN_CHANNELS=32



# Assumes you've built container prior using it, see README
# You can build your own container from NGC 23.10 or 23.11 or earlier 
srun apptainer exec --nv /scratch3/dia021/containers/trchprosthesis_requirements_24.06-py3.sif torchrun --standalone --nnodes=1 --nproc_per_node=4 mnist_ddp.py --epochs=100

