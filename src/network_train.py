'''
network_train.py

This file builds the dataset, dataloader, NN,
and defines the training loop
'''

from torch.optim import optimizer
from pytorch_dataset import SpeechPaceDataset,my_collate_fn
from network_def import SpeechPaceNN
import torch
from torchaudio import transforms,utils
from torch.utils.data import Dataset, DataLoader

'''Training loop function'''
def train_SpeechPaceNN():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    try:
        # hyperparameters
        BATCH_SIZE = 64
        LEARNING_RATE = 0.005
        HIDDEN_SIZE = 64
        NUM_CLASSES = 3
        INPUT_SIZE = 26
        NUM_LAYERS = 1
        NUM_EPOCHS = 2

        # Load the data
        root_dir = "/data/perception-working/Geffen/SpeechPaceData/"
        train_dataset = SpeechPaceDataset(root_dir+"training_data/",root_dir+"training_data/train_labels2.csv")
        val_dataset = SpeechPaceDataset(root_dir+"validation_data/",root_dir+"validation_data/val_labels2.csv")
        test_dataset = SpeechPaceDataset(root_dir+"test_data/",root_dir+"test_data/test_labels2.csv")

        train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,collate_fn=my_collate_fn)
        val_loader = DataLoader(val_dataset,batch_size=BATCH_SIZE,collate_fn=my_collate_fn)
        test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE,collate_fn=my_collate_fn)

        model = SpeechPaceNN(INPUT_SIZE,HIDDEN_SIZE,NUM_LAYERS,NUM_CLASSES)
        model.to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)

        for epoch in range(NUM_EPOCHS):
            for i, (x,lengths,labels) in enumerate(train_loader):
                x,labels = x.to(device),labels.to(device)

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
            eval_model(model,val_loader,device)
            torch.save(model.state_dict(),"../models/model_epoch_"+str(epoch)+".pth")

        print("============================== Finished Training ==============================")
        eval_model(model,val_loader,device)
        torch.save(model.state_dict(),"../models/model_epoch_"+str(epoch)+".pth")



    except KeyboardInterrupt:
        print("============================== QUIT ==============================\n Saving Model ...")
        torch.save(model.state_dict(),"../models/last_model.pth")


# --------------------------------------------------------------------------------------------------------------
        
'''Helper function to evaluate the network (used during training, validation, and testing)'''
def eval_model(model,data_loader,device):
    model.eval()
    model.to(device)

    eval_loss = 0
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()

    predictions = torch.tensor([])
    with torch.no_grad():
        for i, (x,lengths,labels) in enumerate(data_loader):
            x,labels = x.to(device),labels.to(device)
            
            # forward pass
            out = model(x,lengths)

            # look at confusion matrix
            predictions = torch.cat((predictions,out),dim=0)
            gen_conf_mat(predictions,labels)

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

'''Helper function to create a confusion matrix of classification results'''
# this gets called by eval_model with the predictions and labels
def gen_conf_mat(predictions,labels):
    # get the prediction from the max output
    preds = predictions.argmax(dim=1)

    # generate label-prediction pairs
    stacked = torch.stack((labels,preds),dim=1)

    # create the confusion matrix
    conf_mat = torch.zeros(3,3,dtype=torch.int64)

    # fill the confusion matrix
    for pair in stacked:
        x,y = pair.tolist()
        conf_mat[x,y] = conf_mat[x,y]+1

    print(conf_mat)




# --------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    train_SpeechPaceNN()