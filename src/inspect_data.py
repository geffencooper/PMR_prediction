'''
inspect_data.py 

used to check that datasets, dataloaders, and labels
are correct
'''

from pytorch_dataset import SpeechPaceDataset,my_collate_fn,FusedDataset, my_collate_fn_fused
from network_def import SpeechPaceNN,PMRfusionNN
import torch
from torch.utils.data import Dataset, DataLoader
from torchsampler import ImbalancedDatasetSampler
import os
import pandas as pd

root_dir="/data/perception-working/Geffen/avec_data/"
train_labels_csv="binary_sampled_train_metadata2.csv"
val_labels_csv="binary_sampled_val_metadata2.csv"

train_dataset = FusedDataset(root_dir,os.path.join(root_dir,train_labels_csv))
val_dataset = FusedDataset(root_dir,os.path.join(root_dir,val_labels_csv))

print(train_dataset.get_dist())
print(val_dataset.get_dist())

train_loader = DataLoader(train_dataset,32,collate_fn=my_collate_fn_fused,sampler=ImbalancedDatasetSampler(train_dataset))
val_loader = DataLoader(val_dataset,32,collate_fn=my_collate_fn_fused,sampler=ImbalancedDatasetSampler(val_dataset))

for i,(batch) in enumerate(val_loader):
    print(batch[4])
    print(batch[5])
    exit()

# indices = list(range(len(train_dataset)))
# #print("indices:",indices)
# num_samples = len(indices)
# print("num samples:",num_samples)
# df = pd.DataFrame()
# df["label"] = train_dataset.get_labels()
# df.index = indices
# df = df.sort_index()
# print(df)
# label_to_count = df["label"].value_counts()
# print("label to count:",label_to_count)
# weights = 1.0 / label_to_count[df["label"]]
# print("weights:",weights)
# weights = torch.DoubleTensor(weights.to_list())
# print(torch.multinomial(weights,num_samples,replacement=True))