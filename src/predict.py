'''
predict.py

This class is used to try out the network
'''

import torch
from network_def import SpeechPaceNN
import pandas as pd
import numpy as np

class TestNet():
    def __init__(self,model_path):
        self.model = SpeechPaceNN(26,64,1,3)
        self.model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
        self.model.eval()

    def forward(self,sample_path):
        sample = pd.read_csv(sample_path,sep=";")
        sample = sample.iloc[:,np.arange(2,28)]
        features = torch.from_numpy(sample.to_numpy()).float()
        
        output = self.model(features.unsqueeze(0),torch.LongTensor([features.shape[0]]))
        print(torch.argmax(output))


if __name__ =="__main__":
    tn = TestNet("../models/2021-07-30_16-18-53/END_model.pth")
    #tn.forward("C:/Users/gcooper/Desktop/av_feature_extraction/common_voice/aug_train/training_data/sample-000000.csv")
    tn.forward("C:/Users/gcooper/Desktop/av_feature_extraction/PMR_prediction/small_test/data5.csv")