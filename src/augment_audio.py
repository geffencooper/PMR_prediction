'''
augment.py

This script augments the paced audio clips
with noise, reverb, etc for more robust
classification since the original task may
be too easy.
'''

import subprocess
import sounddevice as sd
import soundfile as sf
import time
from audiomentations import Compose, AddGaussianNoise, PitchShift, Shift,TimeStretch
import os
import sox
import numpy as np 

# feature extraction
config_file = "C:/Users/gcooper/Downloads/opensmile-3.0-win-x64/config/mfcc/MFCC12_0_D_A.conf"
output_file = "//totoro/perception-working/Geffen/SpeechPaceData/training_data_aug/"

# create augmentation
augment = Compose([
    AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.02, p=0.5),
    PitchShift(min_semitones=-5, max_semitones=5, p=0.5),
])

#tfm = sox.Transformer()

# get the training files
files = os.listdir("../../common_voice/aug_train")[48280:195780]
#files = os.listdir("../small_test/")
i=0
start = time.time()
# load the wav files and augment them
for f1,f2,f3,f4,f5,f6,f7,f8 in zip(*[iter(files)]*8):
    data1,sr1 = sf.read("../../common_voice/aug_train/"+f1)
    data2,sr2 = sf.read("../../common_voice/aug_train/"+f2)
    data3,sr3 = sf.read("../../common_voice/aug_train/"+f3)
    data4,sr4 = sf.read("../../common_voice/aug_train/"+f4)
    data5,sr5 = sf.read("../../common_voice/aug_train/"+f5)
    data6,sr6 = sf.read("../../common_voice/aug_train/"+f6)
    data7,sr7 = sf.read("../../common_voice/aug_train/"+f7)
    data8,sr8 = sf.read("../../common_voice/aug_train/"+f8)

    data1 = augment(samples=data1,sample_rate=sr1)
    data2 = augment(samples=data2,sample_rate=sr2)
    data3 = augment(samples=data3,sample_rate=sr3)
    data4 = augment(samples=data4,sample_rate=sr4)
    data5 = augment(samples=data5,sample_rate=sr5)
    data6 = augment(samples=data6,sample_rate=sr6)
    data7 = augment(samples=data7,sample_rate=sr7)
    data8 = augment(samples=data8,sample_rate=sr8)

    sf.write("out1.wav",data1,sr1)
    sf.write("out2.wav",data2,sr2)
    sf.write("out3.wav",data3,sr3)
    sf.write("out4.wav",data4,sr4)
    sf.write("out5.wav",data5,sr5)
    sf.write("out6.wav",data6,sr6)
    sf.write("out7.wav",data7,sr7)
    sf.write("out8.wav",data8,sr8)

    dest1 = output_file+f1[:-4]
    dest2 = output_file+f2[:-4]
    dest3 = output_file+f3[:-4]
    dest4 = output_file+f4[:-4]
    dest5 = output_file+f5[:-4]
    dest6 = output_file+f6[:-4]
    dest7 = output_file+f7[:-4]
    dest8 = output_file+f8[:-4]

    procs = [f"""SMILExtract -noconsoleoutput -C {config_file} -I "out1.wav" -O {dest1}.csv""", \
             f"""SMILExtract -noconsoleoutput -C {config_file} -I "out2.wav" -O {dest2}.csv""", \
             f"""SMILExtract -noconsoleoutput -C {config_file} -I "out3.wav" -O {dest3}.csv""", \
             f"""SMILExtract -noconsoleoutput -C {config_file} -I "out4.wav" -O {dest4}.csv""", \
             f"""SMILExtract -noconsoleoutput -C {config_file} -I "out5.wav" -O {dest5}.csv""", \
             f"""SMILExtract -noconsoleoutput -C {config_file} -I "out6.wav" -O {dest6}.csv""", \
             f"""SMILExtract -noconsoleoutput -C {config_file} -I "out7.wav" -O {dest7}.csv""", \
             f"""SMILExtract -noconsoleoutput -C {config_file} -I "out8.wav" -O {dest8}.csv"""]
    curr_procs = []

    for p in procs:
        curr_procs.append(subprocess.Popen(p))
    for p in curr_procs:
        p.wait()
    # i+=4
    # if i % 128 == 0:
    #     print(time.time()-start)

    # tfm.reverb(reverberance=np.random.uniform()*50,high_freq_damping=np.random.uniform()*50)
    # tfm.build(input_array=data, sample_rate_in=sr,output_filepath="../small_test/2"+f)



# data,sr = sf.read("../../common_voice/aug_val/sample-000007.wav")
# sd.play(data,sr)
# time.sleep(len(data)/sr+1)
# sd.stop()

# tfm = sox.Transformer()
# data = augment(samples=data,sample_rate=sr)
# tfm.reverb(reverberance=np.random.uniform()*50,high_freq_damping=np.random.uniform()*50)
# tfm.build(input_array=data, sample_rate_in=sr,output_filepath='../small_test/test.mp3')

# data1 = augment(samples=data, sample_rate=sr)
# data2 = augment(samples=data, sample_rate=sr)
# data3 = augment(samples=data, sample_rate=sr)
# data4 = augment(samples=data, sample_rate=sr)
# data5 = augment(samples=data, sample_rate=sr)

#sf.write("../small_test/test.wav",data,sr)
# sd.play(data,sr)
# time.sleep(len(data)/sr+1)
# sd.stop()
#sf.write("test.wav",data,sr)

# sd.play(data1,sr)
# time.sleep(len(data)/sr+1)
# sd.stop()
# sf.write(file="../small_test/data1.wav",data=data1,samplerate=sr)
# sd.play(data2,sr)
# time.sleep(len(data)/sr+1)
# sd.stop()
# sf.write(file="../small_test/data2.wav",data=data2,samplerate=sr)
# sd.play(data3,sr)
# time.sleep(len(data)/sr+1)
# sd.stop()
# sf.write(file="../small_test/data3.wav",data=data3,samplerate=sr)
# sd.play(data4,sr)
# time.sleep(len(data)/sr+1)
# sd.stop()
# sf.write(file="../small_test/data4.wav",data=data4,samplerate=sr)
# sd.play(data5,sr)
# time.sleep(len(data)/sr+1)
# sd.stop()
# sf.write(file="../small_test/data5.wav",data=data5,samplerate=sr)
