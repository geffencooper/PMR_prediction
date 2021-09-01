'''
time_stretcher.py

this file tries to apply more realistic time stretching to audio
'''

import scipy.io.wavfile as wavfile
import numpy as np

sr, y = wavfile.read("../../avec_data/300_P/300_AUDIO.wav")

def segment(y,num_segments):
    segments = np.empty([100,])
