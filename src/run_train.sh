#!/bin/bash

currDate=$(date +%Y-%m-%d_%H-%M-%S)
mkdir ../models/$currDate

fileName="../models/$1_$currDate/$1.txt"

python -u network_train.py $currDate | tee -i $fileName
