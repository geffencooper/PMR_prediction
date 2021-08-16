#!/bin/bash

# get the current date
currDate=$(date +%Y-%m-%d_%H-%M-%S)

# make this a suffix for the directory name to create a unique directory
dirName=$1_$currDate

# create the directory
mkdir ../models/$dirName

# create the log file
fileName="../models/$1_$currDate/$1.txt"

# execute the python training script with the directory passed as input
# -u means outputs streams are unbuffered, tee sends output streams to a file as well as console,
# -i ensures that ctrl-c will not break the pipe since we want to save the model on ctrl-c and continue logging
python -u network_train.py $dirName | tee -i $fileName
