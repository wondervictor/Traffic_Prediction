#!/usr/bin/env bash

set -e

cfg=trainer_config.py
# pass choice
model="output/pass-00050"
paddle train \
    --config=$cfg \
    --use_gpu=false \
    --job=test \
    --init_model_path=$model \
    --config_args=is_predict=1 \
    --predict_output_dir=.

python gen_result.py > result.csv

rm -rf rank-00000

