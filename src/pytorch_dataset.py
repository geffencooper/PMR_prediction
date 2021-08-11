'''
pytorch_dataset.py

This file contains functions and classes
for creating and loading a pytorch dataset and dataloader
'''


from numpy.lib.polynomial import roots
import torch
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn_utils
import pandas as pd
import numpy as np
import os

'''PyTorch Dataset class for audio pace'''
class SpeechPaceDataset(Dataset):
    # pass in the root dir of the audio data and the path to the labels csv file
    def __init__(self,data_root_dir,labels_csv_path):
        self.root_dir = data_root_dir
        self.labels_frame = pd.read_csv(labels_csv_path,index_col=0)
        self.all_labels = torch.from_numpy(self.labels_frame["label"].values)

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self,idx):
        try:
            # first get the data csv by creating the file name
            data_file = "sample-"
            prefix = ""
            for i in range(6-len(str(idx))):
                prefix += "0"
            data_file += prefix + str(idx) + ".csv"
            data_path = self.root_dir + data_file

            # read the audio features and convert to a tensor
            data_frame = pd.read_csv(data_path,sep=";")
            data_frame = data_frame.iloc[:,np.arange(2,28)]

            # normalize the colummns
            data_frame = (data_frame-data_frame.mean())/data_frame.std()

            features = torch.from_numpy(data_frame.to_numpy())

            # now get the label of this sample as a tensor
            label = torch.tensor(self.labels_frame.at[data_file,"label"])

            return features,label,idx
        except TypeError:
            print("index:",idx)
            print("type:",type(data_frame.to_numpy()))
            print(data_frame)



'''PyTorch Dataset class for audio pace'''
class FusedDataset(Dataset):
    # pass in the root dir of the audio data and the path to the labels csv file
    def __init__(self,data_root_dir,labels_csv_path):
        self.data_root_dir = data_root_dir
        self.labels_frame = pd.read_csv(labels_csv_path) # no index column because using column 0 --> tells us which patient ids in the split
        #self.detailed_labels_frame = pd.read_csv(os.join(data_root_dir,"Detailed_PHQ8_Labels.csv")) # --> tells us the moving subscore
        #self.all_labels = torch.from_numpy(self.labels_frame["label"].values)

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self,idx):
        try:
            # use idx to get the metadata from the labels csv
            patient_id,start,end,label = self.labels_frame.iloc[idx]

            # get the audio features
            audio_features = pd.read_csv(os.path.join(self.data_root_dir,str(int(patient_id))+"_OpenSMILE2.3.0_mfcc.csv"),sep=";",skiprows=int(start*100),nrows=int(end*100))
            audio_features = audio_features.iloc[:,np.arange(2,28)]
            audio_features = torch.from_numpy(audio_features.to_numpy())

            # get the vidual features
            visual_features = pd.read_csv(os.path.join(self.data_root_dir,str(int(patient_id))+"_OpenFace2.1.0_Pose_gaze_AUs.csv"),sep=",",skiprows=int(start*30),nrows=int(end*30))
            visual_features = visual_features.iloc[:,[4,5,6,7,8,9,16,17,26,27]]
            visual_features = torch.from_numpy(visual_features.to_numpy())

            label = torch.tensor(label)

            return audio_features,visual_features,label,idx

            
        except TypeError:
            print("index:",idx)
            print("type:",type(audio_features.to_numpy()))
            print(audio_features)
        

# --------------------------------------------------------------------------------------------------------------
        
'''Helper function to create batches'''
# the batch parameter is a list of (data,label) tuples
def my_collate_fn(batch):
    # sort the tuples in the batch based on the length of the data portion (descending order)
    sorted_batch = sorted(batch,key=lambda x: x[0].shape[0],reverse=True)

    # get the data portion from the batch tuples
    sequences = [x[0] for x in sorted_batch]

    # pad the shorter sequences with zeros
    sequences_padded = rnn_utils.pad_sequence(sequences,batch_first=True)

    # store the true length of the sequences
    lengths = torch.LongTensor([len(x) for x in sequences])

    # get the label portion from the batch tuples
    labels = torch.LongTensor([x[1] for x in sorted_batch])

    # get the index
    idxs = torch.LongTensor([x[2] for x in sorted_batch])

    return sequences_padded.float(),lengths,labels,idxs


# ---------------------------------------------------------------------------
'''Helper function to create batches'''
# the batch parameter is a list of (data,label) tuples
def my_collate_fn_fused(batch):
    # sort the tuples in the batch based on the length of the data portion (descending order)
    audio_sorted_batch = sorted(batch,key=lambda x: x[0].shape[0],reverse=True)
    video_sorted_batch = sorted(batch,key=lambda x: x[1].shape[0],reverse=True)

    # get the data portion from the batch tuples
    audio_sequences = [x[0] for x in audio_sorted_batch]
    video_sequences = [x[1] for x in video_sorted_batch]

    # pad the shorter sequences with zeros
    audio_sequences_padded = rnn_utils.pad_sequence(audio_sequences,batch_first=True)
    video_sequences_padded = rnn_utils.pad_sequence(video_sequences,batch_first=True)

    # store the true length of the sequences
    audio_lengths = torch.LongTensor([len(x) for x in audio_sequences])
    video_lengths = torch.LongTensor([len(x) for x in video_sequences])

    # get the label portion from the batch tuples
    labels = torch.FloatTensor([int(x[2]) for x in audio_sorted_batch])

    # get the index
    idxs = torch.LongTensor([x[3] for x in audio_sorted_batch])

    return audio_sequences_padded.float(),video_sequences_padded.float(),audio_lengths,video_lengths,labels,idxs
