#!/bin/bash

currDate=$(date +%Y-%m-%d_%H:%M:%S)
#mkdir ../models/$currDate
mkdir ../models/base

#fileName="../models/$currDate/$currDate.txt"
fileName="../models/base/$currDate.txt"

python network_train.py | tee $fileName
