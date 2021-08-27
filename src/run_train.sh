#!/bin/bash

# check if config file given as the argument
if [[ $1 == "" ]]; then
    echo "ERROR: no configuration file supplied"
    echo "USAGE: ./run_train.sh ../config_files/[CONFIG_FILE]"
    exit 1
fi

# include the config file variables
source $1

# get the current date
currDate=$(date +%Y-%m-%d_%H-%M-%S)

# make this a suffix for the directory name to create a unique directory
dirName=$session_name-$currDate

# create the directory
mkdir ../models/$dirName
log_dest="../models/$dirName"

# create the log file
fileName="../models/$dirName/$session_name.txt"

# execute the python training script with the directory passed as input and the gpu instance
# -u means outputs streams are unbuffered, tee sends output streams to a file as well as console,
# -i ensures that ctrl-c will not break the pipe since we want to save the model on ctrl-c and continue logging
python -u train_network.py $session_name $log_dest $root_dir --train_data_dir $train_data_dir --val_data_dir $val_data_dir $train_labels_csv \
                           $val_labels_csv $gpu_i $model_name $optim $loss_freq $val_freq \
                           $batch_size $lr $hidden_size $classification $num_classes $regression \
                           $input_size $num_layers $num_epochs $normalize $hidden_init_rand \
                           $weighted_loss $imbalanced_sampler $l2_reg $weight_decay_amnt $dropout $dropout_prob \
                           --load_trained $load_trained --trained_path $trained_path | tee -i $fileName
