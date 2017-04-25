#!/usr/bin/env bash

cd  /home/uniquesc/cth/Traffic_Prediction

# run sh
#
rm -rf output
rm -rf result



host=`hostname`

point_list_file=data/${host}.train.list


echo "${host} initializes data"

# python preprocess.py
echo "${host} starts to train"
# train
sh train.sh ${point_list_file}

# predict
echo "${host} starts to predict"

sh predict.sh ${point_list_file}
