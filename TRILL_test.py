import numpy as np
import librosa
from sklearn import linear_model
import os
import pandas as pd
import sounddevice as sd
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
assert tf.executing_eagerly()

import tensorflow_hub as hub

# load tensorflow model
model = hub.load('https://tfhub.dev/google/nonsemantic-speech-benchmark/trill-distilled/3')

# get patients which have labels
avec_path_prefix = "../avec_data/"
patient_labels = pd.read_csv(avec_path_prefix+"Detailed_PHQ8_Labels.csv")
patient_ids = patient_labels["Participant_ID"].values

# save embeddings
embeddings = pd.DataFrame(index=np.arange(2048),columns=patient_ids)
# embeddings["713"] = np.arange(80000)
# print(embeddings)
# exit()

# calculate embeddings for all patients based on responses
for id in patient_ids:
    print("Patient:",id)

    # load the audio
    y, sr = librosa.load(avec_path_prefix+str(id)+"_P/"+str(id)+"_AUDIO.wav")

    # load the time stamps of responses
    path = os.path.join(avec_path_prefix,
                            (str(id)+"_P/"),
                            (str(id)+"_Transcript.csv"))

    transcript = pd.read_csv(path)

    # get the time stamps as lists
    start_times = transcript["Start_Time"]
    stop_times = transcript["End_Time"]

    # only get the responses
    new_audio = []
    for i,chunk in enumerate(start_times):
        new_audio.extend(y[int(start_times[i]*sr):int(stop_times[i]*sr)])
    new_audio = np.array(new_audio)

    # resample audio to 16K
    REQUIRED_SAMPLE_RATE_ = 16000
    float_audio = new_audio.astype(np.float32) / np.iinfo(np.int16).max
    
    if sr != REQUIRED_SAMPLE_RATE_:
        float_audio = librosa.core.resample(
            float_audio, orig_sr=sr, target_sr=16000, 
            res_type='kaiser_best')
    float_audio = float_audio[:80000] # only use first 5 seconds

    # get embedding and take mean
    emb = model(tf.constant(float_audio,tf.float32), tf.constant(16000,tf.int32))["embedding"]
    emb = np.mean(emb,axis=0)

    # save embedding to data frame
    embeddings[id] = emb

embeddings.to_csv("embeddings.csv")




    
    


# y, sr = librosa.load("../avec_data/300_P/300_AUDIO.wav")

# REQUIRED_SAMPLE_RATE_ = 16000
# float_audio = y.astype(np.float32) / np.iinfo(np.int16).max
# float_audio_16k = []
# if sr != REQUIRED_SAMPLE_RATE_:
#     float_audio = librosa.core.resample(
#         float_audio, orig_sr=sr, target_sr=16000, 
#         res_type='kaiser_best')
# float_audio_16k.append(float_audio)

# tf_out = model(tf.constant(float_audio, tf.float32),tf.constant(16000, tf.int32))['embedding']
# tf_out.shape.assert_is_compatible_with([None, 2048])
# print(tf_out)
# print(tf_out.shape)
