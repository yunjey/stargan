#!/bin/bash
#
#SBATCH --job-name=test_stargan_transient
#SBATCH --output=stargan_transient/res_%j.txt  # output file
#SBATCH -e stargan_transient/res_%j.err        # File to which STDERR will be written
#SBATCH --partition=titanx-long         	   # Partition to submit to
#
#SBATCH --mem=32000                     	   # Memory required in MB
#SBATCH --gres=gpu:1                    	   # No. of required GPUs
#SBATCH --ntasks-per-node=12            	   # No. of cores required
#SBATCH --mem-per-cpu=20000             	   # Memory in MB per cpu allocated

echo "SLURM_JOBID: " $SLURM_JOBID

echo "Start running experiments"

source ~/.bashrc ; source activate tf

python code/main.py --mode test --dataset RaFD --c_dim 40 --rafd_image_dir data/transient_attributes/test --sample_dir stargan_transient/samples --log_dir stargan_transient/logs --model_save_dir stargan_transient/models --result_dir stargan_transient/results

echo "Done"

hostname
sleep 1
exit
