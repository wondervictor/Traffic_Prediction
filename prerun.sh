#!/usr/bin/env bash

mkdir output
mkdir result
touch result.csv
cd data
python split_data.py 10
python preprocess.py
cd ..

pdsh -w node\[1-10\] sh run.sh