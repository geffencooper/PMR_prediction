import torch
import torchaudio
from torchaudio import transforms,utils
from torch.utils.data import Dataset, DataLoader
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

'''PyTorch Dataset class for audio pace'''
class AudioPaceDataset(Dataset):
    def __init__(self,data_csv_path,labels_csv_path):
        self.data_frame = pd.read_csv(data_csv_path)
        self.labels_frame = pd.read_csv(labels_csv_path)

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx=idx.to_list()

        # features is a contiguous set of MFCCs and their deltas
        features = self.data_frame

def get_audio_pace_dataset(train_data_path,val_data_path,test_data_path,train_labels_path,val_labels_path,test_labels_path):
    pass