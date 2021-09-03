# PMR_prediction

This repo contains all the code, configuration files, and trained models for the PMR prediction project

## Important Files
**train_network.py**
* this contains the code for training models
* it should not be changed unless a new model class needs to be added

**network_def.py**
* this is where models get defined
* new models should be appended to this file

**pytorch_dataset.py**
* custom datasets and collate_fn get defined here

## Training a Model
1. First create a configuration file (name.conf) in the config_files directory and copy the contents of skeleton.conf
2. Fill out all the entries with the model class name, training data directory locations, desired hyperparameters, etc
3. run the training script as follows from the src directory: ``` ./run_train.sh ../config_files/[CONFIG FILE NAME]```
