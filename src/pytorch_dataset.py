'''
pytorch_dataset.py

This file contains functions and classes
for creating and loading a pytorch dataset and dataloader
'''


import torch
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn_utils
import pandas as pd
import numpy as np

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
	    #data_frame = (data_frame-data_frame.mean())/data_frame.std()

	    features = torch.from_numpy(data_frame.to_numpy())
            except: TypeError
                print(data_frame)
                print(type(data_frame.to_numpy()))

            # now get the label of this sample as a tensor
            label = torch.tensor(self.labels_frame.at[data_file,"label"])

            return features,label,idx

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
