'''
network_def.py

This file defines the RNN used for training
an speech pace detector
'''


import torch
from torch.cuda import set_device
import torch.nn
import torch.nn.utils.rnn as rnn_utils
import copy 

class SpeechPaceNN(torch.nn.Module):
    def __init__(self,args):
        super(SpeechPaceNN,self).__init__()
        self.args = args

        self.device = torch.device("cuda:"+str(args.gpu_i) if torch.cuda.is_available() else "cpu")

        self.input_size = args.input_size # audio feature length
        self.hidden_size = args.hidden_size # user defined hyperparameter
        self.num_layers = args.num_layers # stacked layers

        # Layer 1: GRU
        self.gru = torch.nn.GRU(args.input_size,args.hidden_size,args.num_layers,batch_first=True)

        # Layer 2: FC for classification/regression
        self.fc = torch.nn.Linear(args.hidden_size,args.num_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight)

        self.dropout = torch.nn.Dropout(args.droput_prob)

        self.init = args.hidden_init_rand
        self.num_classes = args.num_classes

    # initialize the hidden state at the start of each forward pass
    def init_hidden(self,batch_size):
        if self.init == True:
            self.h0 = torch.randn(self.num_layers,batch_size,self.hidden_size)
        else:
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

        if self.args.droput == "y":
            y = self.dropout(y)

        y = self.fc(y)

        # regression
        if self.num_classes == -1:
            return y

        # classification
        else:
            # y = torch.nn.functional.softmax(y,dim=1)
            return y 



# ==================================================================================

class PMRfusionNN(torch.nn.Module):
    def __init__(self,args):
        super(PMRfusionNN,self).__init__()
        self.args = args

        self.device = torch.device("cuda:"+str(args.gpu_i) if torch.cuda.is_available() else "cpu")

        self.input_size = args.input_size # audio feature length
        self.hidden_size = args.hidden_size # user defined hyperparameter
        self.num_layers = args.num_layers # stacked layers

        args_copy = copy.deepcopy(args)
        args_copy.input_size = 26
        args_copy.num_classes = 3
        self.pace_net = SpeechPaceNN(args_copy)
        if args.normalize == "n":
            self.pace_net.load_state_dict(torch.load('../models/speech_pace_RMS_x-2021-08-19_12-29-08/BEST_model.pth',map_location=self.device))
        else:
            self.pace_net.load_state_dict(torch.load('../models/speech_pace_RMS_NORM2-2021-08-24_13-35-00/BEST_model.pth',map_location=self.device))
        
        # fine tune or freeze weights
        for param in self.pace_net.parameters():
            param.requires_grad = False
        

        # Layer 1: GRU for visual features
        self.gru_vis = torch.nn.GRU(args.input_size,args.hidden_size,args.num_layers,batch_first=True)

        # Layer 2: FC for classification/regression after fusion
        self.fc_fusion = torch.nn.Linear(2*args.hidden_size,args.num_classes)

        self.dropout = torch.nn.Dropout(args.droput_prob)

        self.init = args.hidden_init_rand
        self.num_classes = args.num_classes

    # initialize the hidden state at the start of each forward pass
    def init_hidden(self,batch_size):
        if self.init == True:
            self.h0 = torch.randn(self.num_layers,batch_size,self.hidden_size)
        else:
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
        print("audio out:", out)
        y_audio = torch.clone(out[:,-1,:])
        print("y_audio:", y_audio)
        for i,val in enumerate(lengths_audio):
            y_audio[i,:] = out[i,val-1,:]
        print("y_audio:", y_audio)

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
        print("visual out:", out)
        y_visual = torch.clone(out[:,-1,:])
        print("y vis:", y_visual)

        for i,val in enumerate(lengths_visual):
            y_visual[i,:] = out[i,val-1,:]
        print("y vis:", y_visual)

        # ============== concatenate and pass through FC ==============
        
        fused = torch.cat((y_audio,y_visual),dim=1)
        print("fused",fused)

        if self.args.dropout:
            fused = self.dropout(fused)

        y_fused = self.fc_fusion(fused)
        print("final out:",y_fused)
        exit()
        # regression
        if self.num_classes == -1:
            return y_fused

        # classification
        else:
            #y = torch.nn.functional.softmax(y_fused,dim=1)
            return y_fused 