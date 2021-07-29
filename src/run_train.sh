#!/bin/bash

currDate=$(date +%Y-%m-%d_%H:%M:%S)
mkdir $currDate

fileName="../models/$currDate/$currDate.txt"

python test.py $currDate | tee $fileName
