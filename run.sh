#!/usr/bin/env bash

# run sh
#
rm -rf output
rm -rf result

host=`hostname`

point_list_file=${host}.train.list

echo "${host} initializes data"

# python preprocess.py
echo "${host} starts to train"
# train
./train.sh ${point_list_file}

# predict
echo "${host} starts to predict"

./predict.sh ${point_list_file}
