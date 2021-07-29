'''
network_train.py

This file builds the dataset, dataloader, NN,
and defines the training loop and stat helper functions
'''

from torch.optim import optimizer
from pytorch_dataset import SpeechPaceDataset,my_collate_fn
from network_def import SpeechPaceNN
import torch
from torchaudio import transforms,utils
from torch.utils.data import Dataset, DataLoader
import time

'''Training loop function'''
def train_SpeechPaceNN():
    # get the device, hopefully a GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # print model training info
    print("================================ Start Training ================================")
    print("Device:",torch.cuda.current_device()," ----> ",torch.cuda.get_device_name(torch.cuda.current_device()))
    print("Hyperparameters:")

    # hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.002
    HIDDEN_SIZE = 64
    NUM_CLASSES = 3
    INPUT_SIZE = 26
    NUM_LAYERS = 1
    NUM_EPOCHS = 2
    NORMALIZATION = True

    # global variables
    best_val_accuracy = 0
    train_losses = []
    val_losses = []
    val_accuracies = []
    iterations = []
    curr_train_loss = 0
    last_train_loss = 0

    print("Batch Size: {}\nLearning Rate: {}\nHidden Size: {}\nNumber of Layer: {}\n Number of Epochs: {}\n Normalization:{}".format(\
        BATCH_SIZE,LEARNING_RATE,HIDDEN_SIZE,NUM_LAYERS,NUM_EPOCHS,NORMALIZATION))
    try:
        # Load the data
        root_dir = "/data/perception-working/Geffen/SpeechPaceData/"
        train_dataset = SpeechPaceDataset(root_dir+"training_data/",root_dir+"training_data/train_labels2.csv")
        val_dataset = SpeechPaceDataset(root_dir+"validation_data/",root_dir+"validation_data/val_labels2.csv")
        test_dataset = SpeechPaceDataset(root_dir+"test_data/",root_dir+"test_data/test_labels2.csv")

        train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,collate_fn=my_collate_fn)
        val_loader = DataLoader(val_dataset,batch_size=BATCH_SIZE,collate_fn=my_collate_fn)
        test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE,collate_fn=my_collate_fn)

        # build and load the model
        model = SpeechPaceNN(INPUT_SIZE,HIDDEN_SIZE,NUM_LAYERS,NUM_CLASSES)
        model.to(device)

        # loss and optimization criteria
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)

        # track model training time
        start = time.time()

        # ----------------------------------------------------- Training Loop -----------------------------------------------------
        for epoch in range(NUM_EPOCHS):
            # get the next batch
            for i, (x,lengths,labels) in enumerate(train_loader):
                x,labels = x.to(device),labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward pass
                out = model(x,lengths)

                # backward pass
                loss = criterion(out,labels)
                curr_train_loss += loss.item()
                loss.backward()
                optimizer.step()
                
                # print training statistics every n batches
                if i % 20 == 0:
                    print("Train Epoch: {} Iteration: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}".format(epoch,i,i*len(x),len(train_loader.dataset),100.*i/len(train_loader),loss.item()))
                
                # do a validation pass every 10*n batches (lots of training data so don't wait till end of epoch)
                if i % 200 == 0:
                    print("\n----------------- Iteration {} -----------------\n".format(i))

                    # keep track of training and validation loss, since training forward pass takes a while just use accumulated loss for last 10*n batches
                    iterations.append(i)
                    train_losses.append(curr_train_loss-last_train_loss)
                    last_train_loss = curr_train_loss
                    curr_train_loss = 0

                    # validation pass
                    accuracy,val_loss = eval_model(model,val_loader,device)
                    val_accuracies.append(accuracy)
                    val_losses.append(val_loss)

                    # save the most accuracte model up to date
                    if accuracy > best_val_accuracy:
                        best_val_accuracy = accuracy
                        torch.save(model.state_dict(),"../models/best_model"+str(epoch)+".pth")
                    print("Best Accuracy: ",best_val_accuracy,"%")

                    # print the time elapsed
                    end = time.time()
                    elapsed = end-start
                    minutes,seconds = divmod(elapsed,60)
                    hours,minutes = divmod(minutes,60)
                    print("Time Elapsed: {}h {}m {}s".format(hours,minutes,seconds))

            # validation at the end of each epoch, and save model if it is the new best
            accuracy = eval_model(model,val_loader,device)
            if accuracy > best_val_accuracy:
                best_val_accuracy = accuracy
            torch.save(model.state_dict(),"../models/model_epoch_"+str(epoch)+".pth")

        print("================================ Finished Training ================================")
        print("\n----------------- Iteration {} -----------------\n".format(i))
        eval_model(model,val_loader,device)
        torch.save(model.state_dict(),"../models/model_epoch_"+str(epoch)+".pth")
        print("Best Model Val Accuracy:",best_val_accuracy,"%")
        end = time.time()
        elapsed = end-start
        minutes,seconds = divmod(elapsed,60)
        hours,minutes = divmod(minutes,60)
        print("Total Training Time: {}h {}m {}s".format(hours,minutes,seconds))



    except KeyboardInterrupt:
        print("================================ QUIT ================================\n Saving Model ...")
        torch.save(model.state_dict(),"../models/last_model.pth")


# --------------------------------------------------------------------------------------------------------------
        
'''Helper function to evaluate the network (used during training, validation, and testing)'''
def eval_model(model,data_loader,device):
    model.eval()
    model.to(device)

    eval_loss = 0
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()

    all_preds = torch.tensor([])
    all_labels = torch.tensor([],dtype=torch.int)
    all_preds = all_preds.to(device)
    all_labels = all_labels.to(device)
    with torch.no_grad():
        for i, (x,lengths,labels) in enumerate(data_loader):
            x,labels = x.to(device),labels.to(device)
            
            # forward pass
            out = model(x,lengths)

            # accumulate predictions and labels
            all_preds = torch.cat((all_preds,out),dim=0)
            all_labels = torch.cat((all_labels,labels),dim=0)

            # sum up the batch loss
            loss = criterion(out,labels)
            eval_loss += loss.item()

            # get the prediction
            pred = out.max(1,keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()

        gen_conf_mat(all_preds,all_labels)
        #eval_loss /= len(data_loader.dataset)
        print("\nValidation Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(eval_loss,correct,len(data_loader.dataset),100.*correct/len(data_loader.dataset)))

    model.train()
    return 100.*correct/len(data_loader.dataset),eval_loss


# --------------------------------------------------------------------------------------------------------------

'''Helper function to create a confusion matrix of classification results'''
# this gets called by eval_model with the predictions and labels
def gen_conf_mat(predictions,labels):
    # get the prediction from the max output
    preds = predictions.argmax(dim=1)

    # generate label-prediction pairs
    stacked = torch.stack((preds,labels),dim=1)

    # create the confusion matrix
    conf_mat = torch.zeros(3,3,dtype=torch.int64)

    # fill the confusion matrix
    for pair in stacked:
        x,y = pair.tolist()
        conf_mat[x,y] = conf_mat[x,y]+1

    print("\nConfusion Matrix")
    print(conf_mat)
    #print("Correct: ",preds.eq(labels).sum().item())




# --------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    train_SpeechPaceNN()