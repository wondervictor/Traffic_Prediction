#!/usr/bin/env bash

set -e

point_list=$1
# point_list='data/point_count_list_2_tmp'
#output_dir=./output
cfg=trainer_config.py
#   paddle train \
#   --config=$cfg \
#   --save_dir=$output_dir \
#   --trainer_count=2 \
#   --log_period=100 \
#   --dot_period=100 \
#   --num_passes=50 \
#   --use_gpu=false \
#   --config_args=nearby_num=2,subnode=3,point=25380 \
#   --show_parameter_stats_period=3000 \
#   2>&1 | tee 'output/train22670.log'

while read line
do
   point=`echo ${line}|cut -d " " -f 1`
   num=`echo ${line}|cut -d " " -f 2`
   sub_num=`echo ${line}|cut -d " " -f 3`

   output_dir=./output/${point}/
   log=./output/${point}.log
   paddle train \
   --config=${cfg} \
   --save_dir=output \
   --trainer_count=28 \
   --log_period=1000 \
   --dot_period=100 \
   --num_passes=65 \
   --use_gpu=false \
   --config_args=nearby_num=${num},subnode=${sub_num},point=${point} \
   --show_parameter_stats_period=3000 \
   2>&1 | tee ${log}
done < ${point_list}
