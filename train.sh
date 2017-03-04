#!/usr/bin/env bash

set -e

#point_list='data/point_count.txt'
cfg=trainer_config.py
#paddle train \
#  --config=$cfg \
#  --save_dir=./output/1955/ \
#  --trainer_count=2 \
#  --log_period=1000 \
#  --dot_period=100 \
#  --num_passes=50 \
#  --use_gpu=false \
#  --show_parameter_stats_period=3000 \
#  2>&1 | tee 'train.log'
#
#

model=output/1955/pass-00045
paddle train \
   --config=$cfg \
   --use_gpu=false \
   --job=test \
   --init_model_path=$model \
   --config_args=is_predict=1 \
   --predict_output_dir=1955


#
#while read line
#do
#    point=`echo $line|cut -d " " -f 1`
#    num=`echo $line|cut -d " " -f 2`
#    output_dir=./output/$point/
#    paddle train \
#    --config=$cfg \
#    --save_dir=$output_dir \
#    --trainer_count=4 \
#    --log_period=1000 \
#    --dot_period=100 \
#    --num_passes=50 \
#    --use_gpu=false \
#    --config_args=num=$num,point=$point \
#    --show_parameter_stats_period=3000 \
#    2>&1 | tee 'output/train${point}.log'
#
#done < $point_list
#



#for i in { 0..327 }
#do
##paddle train \
##  --config=$cfg \
##  --save_dir=./'output$i' \
##  --trainer_count=32 \
##  --log_period=1000 \
##  --dot_period=100 \
##  --num_passes=1000 \
##  --use_gpu=false \
##  --show_parameter_stats_period=3000 \
##  --config_args=data_id=''
##  2>&1 | tee 'train.log'
#
##    echo $i
#done




