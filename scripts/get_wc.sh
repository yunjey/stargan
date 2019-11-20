#!/bin/bash
#
#SBATCH --job-name=get_wc_data
#SBATCH --output=data/world_cities/res_%j.txt         # output file
#SBATCH -e data/world_cities/res_%j.err               # File to which STDERR will be written
#SBATCH --partition=titanx-long         	   # Partition to submit to
#
#SBATCH --mem=32000                     	   # Memory required in MB
#SBATCH --gres=gpu:1                    	   # No. of required GPUs
#SBATCH --ntasks-per-node=12            	   # No. of cores required
#SBATCH --mem-per-cpu=20000             	   # Memory in MB per cpu allocated

echo "SLURM_JOBID: " $SLURM_JOBID

echo "Start running experiments"

# source ~/.bashrc #; source activate tf

python code/preprocess_worldcities.py

echo "Done"

hostname
sleep 1
exit
