'''
dataset.py

This file contains functions and classes
for preprocessing the audio data
and labels.
'''


import numpy as np
import pyrubberband as pyrb
import random
import sounddevice as sd
import pydub
from pydub.utils import mediainfo
import pandas as pd
import time
import os
import shutil
import glob
    

''' class used to augment audio by time stretching '''
class AugRate():
    # audio can be stretched by random amount between [min_rate, max_rate]
    # ex: audio = 0.9 of original audio
    def __init__(self, sampling_rate, min_rate=None, max_rate=None):
        self.sampling_rate = sampling_rate
        self.min_rate = min_rate
        self.max_rate = max_rate

    # function to slow down audio
    # random factor between [min_rate, 0.9] by default, can also specify factor
    def slow_down(self, audio_clip, factor=None):
        if factor == None:
            factor = random.uniform(self.min_rate,0.9)
        return pyrb.time_stretch(audio_clip, self.sampling_rate, factor),factor

    # function to speed up audio
    # random factor between [1.1, max_rate] by default, can also specify factor
    def speed_up(self, audio_clip, factor=None):
        if factor == None:
            factor = random.uniform(1.1,self.max_rate)
            #print(factor)
        return pyrb.time_stretch(audio_clip, self.sampling_rate, factor),factor

    # helper function to augment a list of files
    # pass in the function for augmenting
    def augment_list(self,src_base_path,dest_base_path,file_list,aug_fn):
        print(dest_base_path)
        l = int(len(file_list)/100)
        # if l == 0:
        #     l = len(file_list)
        #     print(file_list)
        print(l)
        k=-1
        factors = []
        start=time.time()
        for i,f in enumerate(file_list):
            src = src_base_path+f # source path
            dest = dest_base_path+f # destination path

            y,sr = read_mp3(src) # read the file to numpy array
            aug_clip,factor = aug_fn(y)
            factors.append(factor)

            # write to mp3
            original_bitrate = mediainfo(src)['bit_rate']
            aug_clip = np.int16(aug_clip * 2 ** 15)
            audio = pydub.AudioSegment(aug_clip.tobytes(), frame_rate=sr, sample_width=2,channels=1)
            audio.export(dest, format="mp3", bitrate=original_bitrate)
            if i % l == 0:
                end=time.time() 
                k+=1
                print(k,"%","elapsed:",end-start)
        return factors

# --------------------------------------------------------------------------------------------------------------

'''Helper function to read MP3'''
def read_mp3(f):
    y = pydub.AudioSegment.from_mp3(f)
    return np.array(y.get_array_of_samples()),y.frame_rate

# --------------------------------------------------------------------------------------------------------------

'''Helper function to create dataset'''
def create_augmented_dataset(train_path, val_path, test_path):
    # get the list of training, validation, and test audio files
    train_files = os.listdir(train_path)
    val_files = os.listdir(val_path)
    test_files = os.listdir(test_path)

    # get the number of files in each list
    num_train = len(train_files)
    num_val = len(val_files)
    num_test = len(test_files)

    # split into thirds since have 3 classes --> ['normal','sped_up','slowed_down']
    train_split_amnt = num_train//3
    val_split_amnt = num_val//3
    test_split_amnt = num_test//3

    # shuffle the files before augmenting them
    random.shuffle(train_files)
    random.shuffle(val_files)
    random.shuffle(test_files)

    # split the training files into three classes, store files and labels
    normal_train = train_files[:train_split_amnt] # first 3rd
    normal_train_labels = [0]*len(normal_train)
    sped_up_train = train_files[train_split_amnt:2*train_split_amnt] # second 3rd
    sped_up_train_labels = [1]*len(sped_up_train)
    slowed_down_train = train_files[2*train_split_amnt:] # last 3rd
    slowed_down_train_labels = [2]*len(slowed_down_train)

    # split the validation files into three classes, store files and labels
    normal_val = val_files[:val_split_amnt] # first 3rd
    normal_val_labels = [0]*len(normal_val)
    sped_up_val = val_files[val_split_amnt:2*val_split_amnt] # second 3rd
    sped_up_val_labels = [1]*len(sped_up_val)
    slowed_down_val = val_files[2*val_split_amnt:] # last 3rd
    slowed_down_val_labels = [2]*len(slowed_down_val)

    # split the test files into three classes, store files and labels
    normal_test = test_files[:test_split_amnt] # first 3rd
    normal_test_labels = [0]*len(normal_test)
    sped_up_test = test_files[test_split_amnt:2*test_split_amnt] # second 3rd
    sped_up_test_labels = [1]*len(sped_up_test)
    slowed_down_test = test_files[2*test_split_amnt:] # last 3rd
    slowed_down_test_labels = [2]*len(slowed_down_test)

    # create directories for augmented data
    aug_train_path = "../../common_voice/aug_train/"
    aug_val_path = "../../common_voice/aug_val/"
    aug_test_path = "../../common_voice/aug_test/"
    os.mkdir(aug_train_path)
    os.mkdir(aug_val_path)
    os.mkdir(aug_test_path)


    # object used to augment the rate
    ar = AugRate(48000,0.35,1.65)

    # store factors that augment audio by
    train_factors = []
    val_factors = []
    test_factors = []

    # copy normal val
    for i,f in enumerate(normal_val):
        shutil.copy(val_path+f,aug_val_path+f)
        val_factors.append(1)

    # speed up val
    factors = ar.augment_list(val_path,aug_val_path,sped_up_val,ar.speed_up)
    val_factors.extend(factors)
    
    # slow down val
    factors = ar.augment_list(val_path,aug_val_path,slowed_down_val,ar.slow_down)
    val_factors.extend(factors)

    val_labels = pd.DataFrame(data={'file_name' : (normal_val+sped_up_val+slowed_down_val), \
                                    'label' : (normal_val_labels+sped_up_val_labels+slowed_down_val_labels), \
                                    'factor' : val_factors})
    val_labels=val_labels.sample(frac=1)
    val_labels.to_csv("../../common_voice/aug_val/val_labels.csv",index=False)


    # copy normal test
    for i,f in enumerate(normal_test):
        shutil.copy(test_path+f,aug_test_path+f)
        test_factors.append(1)

    # speed up test
    factors = ar.augment_list(test_path,aug_test_path,sped_up_test,ar.speed_up)
    test_factors.extend(factors)
    
    # slow down test
    factors = ar.augment_list(test_path,aug_test_path,slowed_down_test,ar.slow_down)
    test_factors.extend(factors)

    test_labels = pd.DataFrame(data={'file_name' : (normal_test+sped_up_test+slowed_down_test), \
                                     'label' : (normal_test_labels+sped_up_test_labels+slowed_down_test_labels),
                                     'factor' : test_factors})
    test_labels=test_labels.sample(frac=1)
    test_labels.to_csv("../../common_voice/aug_test/test_labels.csv",index=False)


    # copy normal train
    for i,f in enumerate(normal_train):
        shutil.copy(train_path+f,aug_train_path+f)
        train_factors.append(1)

    # speed up train
    factors = ar.augment_list(train_path,aug_train_path,sped_up_train,ar.speed_up)
    train_factors.extend(factors)

    # slow down train
    factors = ar.augment_list(train_path,aug_train_path,slowed_down_train,ar.slow_down)
    train_factors.extend(factors)

    # create a csvs for the labels and shuffle them
    train_labels = pd.DataFrame(data={'file_name' : (normal_train+sped_up_train+slowed_down_train), \
                                      'label' : (normal_train_labels+sped_up_train_labels+slowed_down_train_labels), \
                                      'factor' : train_factors})
    train_labels=train_labels.sample(frac=1)
    train_labels.to_csv("../../common_voice/aug_train/train_labels.csv",index=False)

# --------------------------------------------------------------------------------------------------------------

'''Helper function to convert from mp3 to wav'''
def convert():
    # files                                                                         
    lst = glob.glob("*.mp3")
    #print(lst)
    
    for file in lst:
    # convert wav to mp3
        os.system(f"""ffmpeg -i {file} {file[:-4]}.wav""")

# --------------------------------------------------------------------------------------------------------------

'''Helper function to get features'''
def get_features():
    # files                                                                         
    lst = glob.glob("*.wav")
    config_file = "C:/Users/gcooper/Downloads/opensmile-3.0-win-x64/config/mfcc/MFCC12_0_D_A.conf"

    for file in lst:
        os.system(f"""SMILExtract -C {config_file} -I {file} -O {file[:-4]}.csv""")

# --------------------------------------------------------------------------------------------------------------




if __name__ == "__main__":
    train_path = "../../common_voice/cv-valid-train/cv-valid-train/"
    val_path = "../../common_voice/cv-valid-dev/cv-valid-dev/"
    test_path = "../../common_voice/cv-valid-test/cv-valid-test/"
    create_augmented_dataset(train_path,val_path,test_path)

    '''Test Code'''
    # print("read file")
    # y,sr = read_mp3("../../common_voice/cv-valid-train/cv-valid-train/sample-000000.mp3")
    # print(sr)
    # exit()
    
    # print("augment object")
    # ar = AugRate(sr,0.5,1.5)

    # y_slow = ar.slow_down(y)
    # y_fast = ar.speed_up(y)

    # sd.play(y,sr)
    # time.sleep(6)
    # sd.play(y_slow,sr)
    # time.sleep(6)
    # sd.play(y_fast,sr)
    # time.sleep(6)
    # sd.stop()