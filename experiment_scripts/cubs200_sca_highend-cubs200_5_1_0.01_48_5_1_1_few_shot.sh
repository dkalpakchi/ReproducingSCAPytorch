#!/bin/sh
cd ..
export DATASET_DIR="datasets/"
# Activate the relevant virtual environment:

python train_maml_system.py --name_of_args_json_file experiment_config/cubs200_sca_highend-cubs200_5_1_0.01_48_5_1_1.json --gpu_to_use 0