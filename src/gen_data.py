'''
gen_data.py

This uses SMOTE to add more data to the minority class
'''

from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import os

# fix the random state so samples are aligned (for audio and visual, the same two samples will be used to generate the new sample)
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
    if label > 0:
        label = 1
    else:
        label = 0

    audio_features = pd.read_csv(os.path.join(root_dir,str(int(patient_id))+"_OpenSMILE2.3.0_mfcc.csv"),sep=";",skiprows=int(start*100),nrows=int((end-start)*100))
    audio_features = audio_features.iloc[:,np.arange(2,41)]
    
    visual_features = pd.read_csv(os.path.join(root_dir,str(int(patient_id))+"_OpenFace2.1.0_Pose_gaze_AUs.csv"),sep=",",skiprows=int(start*30),nrows=int((end-start)*30))
    visual_features = visual_features.iloc[:,(np.arange(4,53))]
    
    if len(audio_features) == 500 and len(visual_features) == 150:
        x_audio_train.append(audio_features.to_numpy().flatten())
        x_video_train.append(visual_features.to_numpy().flatten())
        y_train.append(label)
        print(idx)

print("resample")
print(y_train)
print(x_audio_train[0].shape)
x_audio_sm,y_train_sm = sm.fit_resample(x_audio_train,y_train)
x_video_sm,y_train_sm = sm.fit_resample(x_video_train,y_train)

# copy the structure
patient_id,start,end,label = labels_df.iloc[0]
audio_df = pd.read_csv(os.path.join(root_dir,str(int(patient_id))+"_OpenSMILE2.3.0_mfcc.csv"),sep=";")
visual_df = pd.read_csv(os.path.join(root_dir,str(int(patient_id))+"_OpenFace2.1.0_Pose_gaze_AUs.csv"),sep=",")

print("save")
for idx in range(len(y_train)):
    print(idx)
    audio = pd.DataFrame(x_audio_train[idx].reshape(500,39),columns=audio_df.columns[2:])
    visual = pd.DataFrame(x_video_train[idx].reshape(150,49),columns=visual_df.columns[4:])
    
    audio.insert(0,'name',0)
    audio.insert(1,'frameTime',0)
    visual.insert(0,'frame',0)
    visual.insert(1,'timestamp',0)
    visual.insert(2,'confidence',0)
    visual.insert(3,'success',0)
    
    audio.to_csv(root_dir+"SMOTE/"+str(idx)+"_OpenSMILE2.3.0_mfcc",index=False,sep=';')
    visual.to_csv(root_dir+"SMOTE/"+str(idx)+"_OpenFace2.1.0_Pose_gaze_AUs",index=False,sep=',')
labels = pd.DataFrame(y_train_sm)
labels.to_csv(root_dir+"SMOTE/"+"labels")
