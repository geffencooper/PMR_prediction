#!/bin/bash

currDate=$(date +%Y-%m-%d_%H:%M:%S)
fileName="../models/$currDate.txt"

python test.py | tee $fileName
