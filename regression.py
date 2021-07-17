import numpy as np
import librosa
from sklearn import linear_model
import os
import pandas as pd
import random

# read the labels
labels = pd.read_csv("../avec_data/Detailed_PHQ8_Labels.csv")

patients = []
patient_labels = []
patient_scores = []

normal = 0
depressed = 0

classification = True

# remove "intermediate patients"
for index, row in labels.iterrows():
    if classification == True:
        if row["PHQ_8Total"] < 3:
            patients.append(row["Participant_ID"])
            patient_labels.append(0)
            normal+=1
        elif row["PHQ_8Total"] > 9:
            patients.append(row["Participant_ID"])
            patient_labels.append(1)
            depressed+=1
    else:
        patients.append(row["Participant_ID"])
        patient_labels.append(row["PHQ_8Total"])
# print("normal",normal)
# print("depressed",depressed)
# exit()
# get the data
embeddings = pd.read_csv("embeddings.csv")
ids = labels["Participant_ID"].values


# remove the middle patients
to_remove = np.setdiff1d(ids,patients)
for id in to_remove:
    embeddings = embeddings.drop(columns=[str(id)])

# shuffle the data
temp = list(zip(patients,patient_labels))
random.shuffle(temp)
patients,patient_labels = zip(*temp)

# create train and test sets
num_patients = len(patients)
split = int(.85*num_patients)

train_data = list(patients[:split])
train_labels = list(patient_labels[:split])

test_data = list(patients[split:])
test_labels = list(patient_labels[split:])

# print(train_data)
# print(train_labels)
# print(test_data)
# print(test_labels)

# get the data from the ids
for i,id in enumerate(train_data):
    train_data[i] = embeddings[str(id)].values

for i,id in enumerate(test_data):
    test_data[i] = embeddings[str(id)].values

if classification:
    reg = linear_model.LogisticRegression()
else:
    reg = linear_model.LinearRegression()

reg.fit(train_data,train_labels)

print(reg.predict(test_data))
print(test_labels)
print(reg.score(test_data,test_labels))