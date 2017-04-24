#!/usr/bin/env bash

# run sh

rm -rf output
rm -rf result

mkdir output
mkdir result

# python preprocess.py

# train
./train.sh


# predict

./predict.sh

python2.7 data/VadiationSet/RMSEValidator.py data/VadiationSet/419_6_10.csv result.csv