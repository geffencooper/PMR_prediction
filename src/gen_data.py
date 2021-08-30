'''
gen_data.py

This uses SMOTE to add more data to the minority class
'''

from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import os

root_dir = "//totoro/perception-working/Geffen/avec_data/"
train_data_csv = "binary_train_metadata_two_to_one.csv"

labels_df = pd.read_csv(os.path.join(root_dir,train_data_csv))
num_samples = len(labels_df)

x_train = []
y_train = []
for i in range(num_samples):
    
