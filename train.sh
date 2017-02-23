#!/usr/bin/env bash

set -e

cfg=trainer_config.py
paddle train \
  --config=$cfg \
  --save_dir=./output \
  --trainer_count=1 \
  --log_period=100 \
  --dot_period=10 \
  --num_passes=10 \
  --use_gpu=false \
  --show_parameter_stats_period=3000 \
  2>&1 | tee 'train.log'
