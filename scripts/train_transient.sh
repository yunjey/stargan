#!/bin/bash
#
#SBATCH --job-name=train_stargan
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

save_path="/mnt/nfs/work1/jensen/aatrey/SaliencyMaps/toybox/ctoybox/models/AmidarToyboxNoFrameskip-v4/amidar4e7_a2c.model"
log_dir="/mnt/nfs/work1/jensen/aatrey/SaliencyMaps/toybox/ctoybox/models/AmidarToyboxNoFrameskip-v4/logs/amidar4e7_a2c"

python code/main.py --mode train --dataset RaFD --c_dim 40 --rafd_image_dir data/transient_attributes/train --sample_dir stargan_transient/samples --log_dir stargan_transient/logs --model_save_dir stargan_transient/models --result_dir stargan_transient/results

# OPENAI_LOGDIR=$log_dir python -m baselines.run --alg=a2c --env=AmidarToyboxNoFrameskip-v4 --num_timesteps=4e7 --save_path=$save_path

echo "Done"

hostname
sleep 1
exit

#test using : ./start_python -m baselines.run --alg=a2c --env=AmidarToyboxNoFrameskip-v4 --num_timesteps=0 --load_path=./models/AmidarToyboxNoFrameskip-v4/amidar_a2c.model --num_env=1 --play
