#!/usr/bin/env bash

set -e

cfg=trainer_config.py
# pass choice
point_list='data/point_count.txt'

while read line
do
    point=`echo $line|cut -d " " -f 1`
    num=`echo $line|cut -d " " -f 2`
    model=output/$point/pass-00150

    paddle train \
        --config=$cfg \
        --use_gpu=false \
        --job=test \
        --init_model_path=$model \
        --config_args=is_predict=1 \
        --config_args=num=$num,point=$point \
        --predict_output_dir=./$point
    #rm -rf rank-00000

done < $point_list

