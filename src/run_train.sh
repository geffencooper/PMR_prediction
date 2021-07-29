#!/bin/bash

currDate=$(date +%Y-%m-%d)
fileName="$currDate.txt"

python network_train.py | tee fileName