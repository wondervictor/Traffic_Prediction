#!/usr/bin/env bash

set -e

cfg=trainer_config.py
# pass choice
    model=output/test/pass-00049
    paddle train \
        --config=$cfg \
        --use_gpu=false \
        --job=test \
        --init_model_path=$model \
        --config_args=is_predict=1,nearby_num=2,subnode=6,point=22670 \
        --predict_output_dir=result/22670
        python2.7 generate_result.py 22670 result/22670/rank-00000

#while read line
#do
#    point=`echo $line|cut -d " " -f 1`
#    num=`echo $line|cut -d " " -f 2`
#    model=output/$point/pass-00049
#    paddle train \
#        --config=$cfg \
#        --use_gpu=false \
#        --job=test \
#        --init_model_path=$model \
#        --config_args=is_predict=1,num=$num,point=$point \
#        --predict_output_dir=result/$point
#        python2.7 generate_result.py $point result/${point}/rank-00000
#
#done < $point_list
