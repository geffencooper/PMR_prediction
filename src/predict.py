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
from train_network import parse_args,eval_model


if __name__ =="__main__":
    args = parse_args()

    device = torch.device("cuda:"+str(args.gpu_i) if torch.cuda.is_available() else "cpu")

    val_dataset = FusedDataset("//totoro/perception-working/Geffen/avec_data/","binary_val_metadata.csv")
    val_loader = DataLoader(val_dataset,32,collate_fn=my_collate_fn_fused)#,sampler=ImbalancedDatasetSampler(val_dataset))

    pmr = PMRfusionNN(23,64,1,2,2)
    pmr.load_state_dict(torch.load("../models/PMR_fusion-2021-08-20_12-19-49/BEST_model.pth"))

    pmr.eval()
    pmr.to(device)

    eval_model(pmr,val_loader,device,torch.nn.CrossEntropyLoss,args)