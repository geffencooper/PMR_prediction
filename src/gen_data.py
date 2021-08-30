'''
gen_data.py

This uses SMOTE to add more data to the minority class
'''

from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import os

sm = SMOTE(random_state=42)

root_dir = "/data/perception-working/Geffen/avec_data/"
train_data_csv = "binary_train_metadata_two_to_one.csv"

labels_df = pd.read_csv(os.path.join(root_dir,train_data_csv))
num_samples = len(labels_df)

x_audio_train = []
x_video_train = []
y_train = []
for idx in range(num_samples):
    patient_id,start,end,label = labels_df.iloc[idx]

    #print("read audio ",idx)
    audio_features = pd.read_csv(os.path.join(root_dir,str(int(patient_id))+"_OpenSMILE2.3.0_mfcc.csv"),sep=";",skiprows=int(start*100),nrows=int((end-start)*100))
    audio_features = audio_features.iloc[:,np.arange(2,28)]
    audio_features = (audio_features-audio_features.mean())/audio_features.std()
    print(len(audio_features))
    #print("read video ",idx)
    visual_features = pd.read_csv(os.path.join(root_dir,str(int(patient_id))+"_OpenFace2.1.0_Pose_gaze_AUs.csv"),sep=",",skiprows=int(start*30),nrows=int((end-start)*30))
    visual_features = visual_features.iloc[:,np.concatenate((np.arange(4,10),np.arange(18,35)))]
    visual_features = (visual_features-visual_features.mean())/(visual_features.std() + 0.01)
    print(len(visual_features))
    x_audio_train.append(audio_features.to_numpy())
    x_video_train.append(visual_features.to_numpy())
    y_train.append(int(label))

# x_audio_sm,y_train_sm = sm.fit_resample(x_audio_train,y_train)
# x_video_sm,y_train_sm = sm.fit_resample(x_video_train,y_train)
