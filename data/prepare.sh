#!/usr/bin/env bash
mkdir ../ouput
mkdir speed_data
mkdir predict_data
touch train.list
touch test.list
touch pred.list

python2.7 preprocess.py
