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
3. Run the training script as follows from the src directory: ``` ./run_train.sh ../config_files/[CONFIG FILE NAME]```
4. The status and results of training will be printed to the screen and logged to a new directory that has the current date and time appended to it
5. The training script saves the model with the highest validation accuracy, as well as the last model after all epochs are completed
6. If the training script is exited in the middle (ctrl+c), it will evaluate the current model and save it

## Testing a Model
1. Run the predict script as follows from the src directory: ``` ./run_predict.sh ../config_files/[CONFIG FILE NAME]``` (this is the same configuration file used for training)
2. Before you do this:
  * fill in the filename for the validation dataset labels in predict.py, the dataset specified in the configuration file is what is used for model evaluation
  * fill in the path to the model you want to test, this is found in the models directory

## Comparing Models
The rank.py script will parse all the log files and rank the models by class and best validation accuracy.

## Other files
There are other scripts in the src folder that were used to preprocess data. For example, gen_data.py was used to generate artificial samples using SMOTE.
