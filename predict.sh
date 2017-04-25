#!/usr/bin/env bash

set -e

point_list=$1

cfg=trainer_config.py
#model=output/pass-00049
#    paddle train \
#        --config=$cfg \
#        --use_gpu=false \
#        --job=test \
#        --init_model_path=$model \
#        --config_args=is_predict=1,nearby_num=2,subnode=3,point=25380 \
#        --predict_output_dir=result/25380
#        python2.7 generate_result.py 25380 result/25380/rank-00000

#point_list='data/point_count_list_2_tmp'

python csv_timestamp.py result.csv 201604190800 5 24

while read line
do
    point=`echo ${line}|cut -d " " -f 1`
    num=`echo ${line}|cut -d " " -f 2`
    sub_num=`echo ${line}|cut -d " " -f 3`
    model=output/${point}/pass-00049
    paddle train \
        --config=${cfg} \
        --use_gpu=false \
        --job=test \
        --init_model_path=${model} \
        --config_args=is_predict=1,nearby_num=${num},subnode=${sub_num},point=${point} \
        --predict_output_dir=result/${point}

done < ${point_list}
