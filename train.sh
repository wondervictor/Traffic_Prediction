#!/usr/bin/env bash

set -e

cfg=trainer_config.py
paddle train \
  --config=$cfg \
  --save_dir=./output \
  --trainer_count=32 \
  --log_period=1000 \
  --dot_period=100 \
  --num_passes=1000 \
  --use_gpu=false \
  --show_parameter_stats_period=3000 \
  2>&1 | tee 'train.log'
