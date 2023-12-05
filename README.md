# PyTorch multi-GPU and multi-node examples for CSC's supercomputers

These examples are based on the repository: https://github.com/CSCfi/pytorch-ddp-examples/tree/master
tailored for our HPC needs (anonymised where appropriate). 
## Build container     

```shell
export APPTAINER_CACHEDIR='/scratch3/$USERNAME/.singularity'
export APPTAINER_TMPDIR='/scratch3/$USERNAME/temp'
module load apptainer # singularity 
    
mkdir /scratch3/$USERNAME/containers
cd /scratch3/$USERNAME/containers
apptainer build trchprosthesis_requirements:23.11-py3.sif docker://fdiakogiannis/trchprosthesis_requirements:23.11-py3
```

## Example run:     
```shell
sbatch -A YourProjectCode run-ddp-torchrun-Multinode.sh
```
