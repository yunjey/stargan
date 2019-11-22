#!/bin/bash
#
#SBATCH --job-name=train_stargan_seasonal
#SBATCH --output=stargan_seasonal/res_%j.txt  # output file
#SBATCH -e stargan_seasonal/res_%j.err        # File to which STDERR will be written
#SBATCH --partition=titanx-long         	   # Partition to submit to
#
#SBATCH --mem=32000                     	   # Memory required in MB
#SBATCH --gres=gpu:1                    	   # No. of required GPUs
#SBATCH --ntasks-per-node=12            	   # No. of cores required
#SBATCH --mem-per-cpu=20000             	   # Memory in MB per cpu allocated

echo "SLURM_JOBID: " $SLURM_JOBID

echo "Start running experiments"

# source ~/.bashrc ; source activate tf

python3 code/main.py --mode train --dataset RaFD --c_dim 14 --rafd_image_dir data/transient_attributes/train --sample_dir stargan_seasonal/samples \
		 --log_dir stargan_seasonal/logs --model_save_dir stargan_seasonal/models --result_dir stargan_seasonal/results \
		 --selected_attrs daylight night sunrisesunset dawndusk sunny clouds fog storm snow spring summer autumn winter rain

echo "Done"

hostname
sleep 1
exit
