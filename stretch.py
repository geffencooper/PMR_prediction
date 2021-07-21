import numpy as np
import librosa
from numpy import lib
from sklearn import linear_model
import os
import pandas as pd
import random
import sounddevice as sd
import time
import pytsmod as tsm
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T



# y,sr = torchaudio.load("../avec_data/306_P/306_AUDIO.wav")
# tsfm = torchaudio.transforms.TimeStretch(fixed_rate=0.5)
# y=tsfm(y)
# # y_new = y[int(435*sr):int(445*sr)]
# # sf.write('out.wav',y_new,sr)
# exit()
# y_new = librosa.effects.time_stretch(y_new,1.3)

# print("play")
# sd.play(y,sr)
# time.sleep(7)
# sd.stop()

# sd.play(y_new,sr)
# time.sleep(7)
# sd.stop()