#!/bin/sh
cd ..
export DATASET_DIR="datasets/"
# Activate the relevant virtual environment:

python train_maml_system.py --name_of_args_json_file experiment_config/cubs200_sca-cubs200_1_2_0.01_48_5_5_1.json --gpu_to_use 0