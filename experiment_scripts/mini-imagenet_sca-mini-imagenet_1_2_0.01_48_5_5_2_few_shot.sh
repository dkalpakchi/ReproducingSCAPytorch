#!/bin/sh
cd ..
export DATASET_DIR="datasets/"
# Activate the relevant virtual environment:

python train_maml_system.py --name_of_args_json_file experiment_config/mini-imagenet_sca-mini-imagenet_1_2_0.01_48_5_5_2.json --gpu_to_use 0