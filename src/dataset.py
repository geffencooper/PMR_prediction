'''
dataset.py

This file contains functions and classes
for preprocessing and loading the audio data
and labels.
'''

import pydub
import numpy as np
import torch
import torchaudio
from torchaudio import transforms,utils
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pyrubberband as pyrb
import random
import sounddevice as sd
import time

# =====================================================================
# =========================== Preprocessing ===========================
# =====================================================================

'''Helper Function to read MP3'''
import pydub 
import numpy as np

def read_mp3(f):
    y = pydub.AudioSegment.from_mp3(f)
    return np.array(y.get_array_of_samples()),y.frame_rate
    

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
            print(factor)
        return pyrb.time_stretch(audio_clip, self.sampling_rate, factor)

    # function to speed up audio
    # random factor between [1.1, max_rate] by default, can also specify factor
    def speed_up(self, audio_clip, factor=None):
        if factor == None:
            factor = random.uniform(1.1,self.max_rate)
            print(factor)
        return pyrb.time_stretch(audio_clip, self.sampling_rate, factor)


if __name__ == "__main__":
    print("read file")
    y,sr = read_mp3("../../common_voice/cv-valid-train/cv-valid-train/sample-000000.mp3")
    
    print("augment object")
    ar = AugRate(sr,0.5,1.5)

    y_slow = ar.slow_down(y)
    y_fast = ar.speed_up(y)

    sd.play(y,sr)
    time.sleep(6)
    sd.play(y_slow,sr)
    time.sleep(6)
    sd.play(y_fast,sr)
    time.sleep(6)
    sd.stop()

# def download_dataset():
#     # define the speed transformation
#     transform = transforms.Compose(
#         [transforms.ToTensor()])

#     # first get the audio dataset
#     train_set = torchaudio.datasets.COMMONVOICE(root='./data',train=True,download=True,transform=transform)

# class AudioSpeedDataset(Dataset):
#     def __init__(self, avec_path, root_dir, transform=None):
#         self.landmarks_frame = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#         self.transform = transform
        
#     def __len__(self):
#         return len(self.landmarks_frame)
    
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#         # the landmarks dataset has 4 coordinates (8 columns)
#         # the img name is the last column
#         img_name = os.path.join(self.root_dir,self.landmarks_frame.iloc[idx, 8])
#         image = io.imread(img_name)
#         landmarks = self.landmarks_frame.iloc[idx,0:4] # only eyes for now
#         landmarks = (np.array(landmarks).astype('float'))
        
#         # sanity check
#         # plt.imshow(image, cmap='gray')
#         # plt.scatter((landmarks[0],landmarks[2]), (landmarks[1],landmarks[3]), c='r', s=40) 
#         # plt.show()
        
#         image = (torch.Tensor(image).float())#/255
#         image = image.unsqueeze(0) # add a dimension for grayscale
#         landmarks = torch.from_numpy(landmarks).float()
        
        
#         if self.transform is not None:
#             image = self.transform(image)
            
#         return image, landmarks