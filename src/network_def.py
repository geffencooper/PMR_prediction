'''
network_def.py

This file defines the RNN used for training
an speech pace detector
'''


import torch
import torch.nn
import torch.nn.utils.rnn as rnn_utils

class SpeechPaceNN(torch.nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,num_classes):
        super(SpeechPaceNN,self).__init__()
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.input_size = input_size # audio feature length
        self.hidden_size = hidden_size # user defined hyperparameter
        self.num_layers = num_layers # stacked layers

        # Layer 1: GRU
        self.gru = torch.nn.GRU(input_size,hidden_size,num_layers,batch_first=True)

        # Layer 2: FC for classification/regression
        self.fc = torch.nn.Linear(hidden_size,num_classes)

    # initialize the hidden state at the start of each forward pass
    def init_hidden(self,batch_size):
        #self.h0 = torch.randn(self.num_layers,batch_size,self.hidden_size)
        self.h0 = torch.zeros(self.num_layers,batch_size,self.hidden_size)
        self.h0 = self.h0.to(self.device)

    def forward(self,X,lengths):
        batch_size,sequence_len,feature_len = X.size()

        # initialize the hidden state
        self.init_hidden(batch_size)

        # pack the padded sequence so the network ignores the zeros
        X = rnn_utils.pack_padded_sequence(X,lengths,batch_first=True)

        # pass through GRU
        out,hidden = self.gru(X,self.h0)

        # undo the packing
        out, _ = rnn_utils.pad_packed_sequence(out,batch_first=True)

        # out is [batch_size, seq_len, hidden_size]
        # We need to flatten the input for the FC layer
        # We also only want the last time step of the output since we are doing a many-to-1 RNN
        y = torch.clone(out[:,-1,:])

        for i,val in enumerate(lengths):
            y[i,:] = out[i,val-1,:]

        y = self.fc(y)

        y = torch.nn.functional.softmax(y,dim=1)

        return y 



# ==================================================================================

class PMRfusionNN(torch.nn.Module):
    def __init__(self,input_size,hidden_size,num_layers):
        super(PMRfusionNN,self).__init__()
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.input_size = input_size # audio feature length
        self.hidden_size = hidden_size # user defined hyperparameter
        self.num_layers = num_layers # stacked layers

        self.pace_net = SpeechPaceNN(26,64,1,3)
        self.pace_net.load_state_dict(torch.load('../models/2021-08-06_17-21-36/BEST_model.pth'))

        for param in self.pace_net.parameters():
            param.requires_grad = False
        

        # Layer 1: GRU for visual features
        self.gru_vis = torch.nn.GRU(input_size,hidden_size,num_layers,batch_first=True)

        # Layer 2: FC for classification/regression after fusion
        self.fc_fusion = torch.nn.Linear(2*hidden_size,1)

    # initialize the hidden state at the start of each forward pass
    def init_hidden(self,batch_size):
        #self.h0 = torch.randn(self.num_layers,batch_size,self.hidden_size)
        self.h0 = torch.zeros(self.num_layers,batch_size,self.hidden_size)
        self.h0 = self.h0.to(self.device)

    def forward(self,X_audio,lengths_audio,X_visual,lengths_visual):
        # ==================== first pass through audio portion ====================
        batch_size,sequence_len,feature_len = X_audio.size()

        # initialize the hidden state
        self.pace_net.init_hidden(batch_size)

        # pack the padded sequence so the network ignores the zeros
        X_audio = rnn_utils.pack_padded_sequence(X_audio,lengths_audio,batch_first=True)

        # pass through GRU
        out,hidden = self.pace_net.gru(X_audio,self.pace_net.h0)

        # undo the packing
        out, _ = rnn_utils.pad_packed_sequence(out,batch_first=True)

        # out is [batch_size, seq_len, hidden_size]
        # We need to flatten the input for the FC layer
        # We also only want the last time step of the output since we are doing a many-to-1 RNN
        y_audio = torch.clone(out[:,-1,:])

        for i,val in enumerate(lengths_audio):
            y_audio[i,:] = out[i,val-1,:]

        # ==================== then pass through visual portion ====================
        batch_size,sequence_len,feature_len = X_visual.size()

        # initialize the hidden state
        self.init_hidden(batch_size)

        # pack the padded sequence so the network ignores the zeros
        X_visual = rnn_utils.pack_padded_sequence(X_visual,lengths_visual,batch_first=True)

        # pass through GRU
        out,hidden = self.gru_vis(X_visual,self.h0)

        # undo the packing
        out, _ = rnn_utils.pad_packed_sequence(out,batch_first=True)

        # out is [batch_size, seq_len, hidden_size]
        # We need to flatten the input for the FC layer
        # We also only want the last time step of the output since we are doing a many-to-1 RNN
        y_visual = torch.clone(out[:,-1,:])

        for i,val in enumerate(lengths_visual):
            y_visual[i,:] = out[i,val-1,:]

        # ============== concatenate and pass through FC ==============

        y_fused = self.fc_fusion(torch.cat((y_audio,y_visual),dim=0))

        return y_fused 