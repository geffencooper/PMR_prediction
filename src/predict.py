'''
predict.py

This class is used to try out the network
'''

import torch
from network_def import PMRfusionNN, SpeechPaceNN
from pytorch_dataset import FusedDataset,my_collate_fn_fused
import pandas as pd
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torchsampler import ImbalancedDatasetSampler
from train_network import train_nn
from train_network import parse_args,eval_model,create_dataset,create_loader,create_model
import copy

if __name__ =="__main__":
    args = parse_args()

    device = torch.device("cuda:"+str(args.gpu_i) if torch.cuda.is_available() else "cpu")

    val_dataset = create_dataset(args,args.val_labels_csv,args.val_data_dir)
    val_loader = create_loader(val_dataset,args)

    pmr = PMRfusionNN(args)
    pmr.load_state_dict(torch.load("../models/binary_fusion_one_to_one_NORM-2021-08-27_13-15-21/BEST_model.pth"))

    pmr.eval()
    pmr.to(device)

    eval_model(pmr,val_loader,device,torch.nn.CrossEntropyLoss(),args)