'''
calc_deltas.py

This script is used to get the delta of the
visual features
'''

import python_speech_features as psf
import os
import pandas as pd
import numpy as np

base_path = "//totoro/perception-working/Geffen/avec_data/"
dest_suffix = "_deltas.csv"

# use the pose and AUs
feature_idxs = ["pose_Tx","pose_Ty","pose_Tz","pose_Rx","pose_Ry","pose_Rz","AU01_r","AU02_r","AU04_r","AU05_r","AU06_r","AU07_r","AU09_r", \
                "AU10_r","AU12_r","AU14_r","AU15_r","AU17_r","AU20_r","AU23_r","AU25_r","AU26_r","AU45_r"]

files = os.listdir(base_path)

for file in files:
    if file.endswith("AUs.csv"):
        print(file)
        frame = pd.read_csv(base_path+file)
        delta_frame = psf.base.delta(frame[feature_idxs],2)
        delta_frame = pd.DataFrame(delta_frame,columns=feature_idxs)
        delta_frame.insert(0,"timestamp",frame["timestamp"])
        delta_frame.to_csv(base_path+file[:-4]+dest_suffix,index=False)
