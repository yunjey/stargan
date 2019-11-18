#!/bin/bash
#
#SBATCH --job-name=train_stargan_landmarks
#SBATCH --output=stargan_landmarks/res_%j.txt  # output file
#SBATCH -e stargan_landmarks/res_%j.err        # File to which STDERR will be written
#SBATCH --partition=titanx-long         	   # Partition to submit to
#
#SBATCH --mem=32000                     	   # Memory required in MB
#SBATCH --gres=gpu:1                    	   # No. of required GPUs
#SBATCH --ntasks-per-node=12            	   # No. of cores required
#SBATCH --mem-per-cpu=20000             	   # Memory in MB per cpu allocated

echo "SLURM_JOBID: " $SLURM_JOBID

echo "Start running experiments"

source ~/.bashrc ; source activate tf

python code/main.py --mode train --dataset RaFD --c_dim 20 --rafd_image_dir data/landmarks/train --sample_dir stargan_landmarks/samples --log_dir stargan_landmarks/logs --model_save_dir stargan_landmarks/models --result_dir stargan_landmarks/results

echo "Done"

hostname
sleep 1
exit
