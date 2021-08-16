#!/bin/bash

currDate=$(date +%Y-%m-%d_%H-%M-%S)
mkdir ../models/$currDate

fileName="../models/$currDate/$1.txt"

python -u network_train.py $currDate | tee -i $fileName
