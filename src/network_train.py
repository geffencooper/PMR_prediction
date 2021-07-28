'''
network_train.py

This file builds the dataset, dataloader, NN,
and defines the training loop
'''

from torch.optim import optimizer
from pytorch_dataset import SpeechPaceDataset,my_collate_fn
from network_def import SpeechPaceNN
import torch
import torchaudio
from torchaudio import transforms,utils
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
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


def train_SpeechPaceNN():
    try:
        # hyperparameters
        BATCH_SIZE = 4
        LEARNING_RATE = 0.001
        HIDDEN_SIZE = 128
        NUM_CLASSES = 3
        INPUT_SIZE = 26
        NUM_LAYERS = 1
        NUM_EPOCHS = 2

        # Load the data
        root_dir = "../../common_voice/"
        train_dataset = SpeechPaceDataset(root_dir+"aug_train/",root_dir+"aug_train/train_labels2.csv")
        val_dataset = SpeechPaceDataset(root_dir+"aug_val/",root_dir+"aug_val/val_labels2.csv")
        test_dataset = SpeechPaceDataset(root_dir+"aug_test/",root_dir+"aug_test/test_labels2.csv")

        train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,collate_fn=my_collate_fn)
        val_loader = DataLoader(val_dataset,batch_size=BATCH_SIZE,collate_fn=my_collate_fn)
        test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE,collate_fn=my_collate_fn)

        model = SpeechPaceNN(INPUT_SIZE,HIDDEN_SIZE,NUM_LAYERS,NUM_CLASSES)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)

        for epoch in range(NUM_EPOCHS):
            running_loss = 0.0
            for i, (x,lengths,labels) in enumerate(train_loader):

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                out = model(x,lengths)

                # backward
                loss = criterion(out,labels)
                loss.backward()
                optimizer.step()

                
                # print statistics after each iteration
                print("Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}".format(epoch,i*len(x),len(train_loader.dataset),100.*i/len(train_loader),loss.item()))

            # validation at the end of each epoch, and save model
            eval_model(model,val_dataset)
            torch.save(model.state_dict(),"../models/model_epoch_"+str(epoch)+".pth")

        print("============================== Finished Training ==============================")
        eval_model(model,val_dataset)
        torch.save(model.state_dict(),"../models/model_epoch_"+str(epoch)+".pth")



    except KeyboardInterrupt:
        print("============================== QUIT ==============================\n Saving Model ...")
        torch.save(model.state_dict(),"../models/last_model.pth")


# --------------------------------------------------------------------------------------------------------------
        
'''Helper function to evaluate the network (used during training, validation, and testing)'''
def eval_model(model,data_loader):
    model.eval()

    eval_loss = 0
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for i, (x,lengths,labels) in enumerate(data_loader):
            out = model(x)

            # sum up the batch loss
            loss = criterion(out,x)
            eval_loss += loss.item()

            # get the prediction
            pred = out.max(1,keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()

        eval_loss /= len(data_loader.dataset)
        print("\n Average Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(eval_loss,correct,len(data_loader.dataset),100.*correct/len(data_loader.dataset)))

    # model.train()

# --------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    train_SpeechPaceNN()