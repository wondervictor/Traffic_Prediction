#!/usr/bin/env bash

set -e

cfg=trainer_config.py
model=output/pass-00049
    paddle train \
        --config=$cfg \
        --use_gpu=false \
        --job=test \
        --init_model_path=$model \
        --config_args=is_predict=1,nearby_num=4,subnode=5,point=28469 \
        --predict_output_dir=result/28469
        python2.7 generate_result.py 28126 result/28469/rank-00000

#point_list='data/point_count_list_2'
#
#while read line
#do
#    point=`echo ${line}|cut -d " " -f 1`
#    num=`echo ${line}|cut -d " " -f 2`
#    sub_num=`echo ${line}|cut -d " " -f 2`
#    model=output/${point}/pass-00049
#    paddle train \
#        --config=${cfg} \
#        --use_gpu=false \
#        --job=test \
#        --init_model_path=${model} \
#        --config_args=nearby_num=${num},subnode=${sub_num},point=${point} \
#        --predict_output_dir=result/${point}
#        python2.7 generate_result.py ${point} result/${point}/rank-00000
#
#done < ${point_list}
