#!/bin/bash

currDate=$(date +%Y-%m-%d_%H:%M:%S)
mkdir ../models/$currDate

fileName="../models/$currDate/$currDate.txt"

python -u network_train.py $currDate | tee $fileName
