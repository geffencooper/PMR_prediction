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
import sys

'''Training loop function'''
def train_SpeechPaceNN():
    # get the device, hopefully a GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # print model training info
    print("\n\n\n================================ Start Training ================================")
    print("\nDevice:",torch.cuda.current_device()," ----> ",torch.cuda.get_device_name(torch.cuda.current_device()))
    print("\nHyperparameters:")

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
    output_location=""
    best_val_accuracy = 0
    train_losses = []
    val_losses = []
    val_accuracies = []
    iterations = []
    curr_train_loss = 0
    last_train_loss = 0

    print("Batch Size: {}\nLearning Rate: {}\nHidden Size: {}\nNumber of Layer: {}\nNumber of Epochs: {}\nNormalization:{}".format(\
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
                if i % 20 == 0 and i != 0:
                    print("Train Epoch: {} Iteration: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}".format(epoch,i,i*len(x),len(train_loader.dataset),100.*i/len(train_loader),loss.item()))
                
                # do a validation pass every 10*n batches (lots of training data so don't wait till end of epoch)
                if i % 200 == 0:
                    print("\n\n----------------- Epoch {} Iteration {} -----------------\n".format(epoch,i))

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
                        torch.save(model.state_dict(),"../models/"+str(output_location)+"/BEST_model.pth")
                    print("Best Accuracy: ",best_val_accuracy,"%")

                    # print the time elapsed
                    end = time.time()
                    elapsed = end-start
                    minutes,seconds = divmod(elapsed,60)
                    hours,minutes = divmod(minutes,60)
                    print("Time Elapsed: {}h {}m {}s".format(int(hours),int(minutes),int(seconds)))
                    print("\n-----------------------------------------------\n\n")

        print("================================ Finished Training ================================")
        print("\n----------------- Epoch {} Iteration {} -----------------\n".format(epoch,i))
        torch.save(model.state_dict(),"../models/END_model_epoch_"+str(epoch)+".pth")
        
        # validation pass
        accuracy,val_loss = eval_model(model,val_loader,device)
        val_accuracies.append(accuracy)
        val_losses.append(val_loss)

        # save the most accuracte model up to date
        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            torch.save(model.state_dict(),"../models/"+str(output_location)+"/BEST_model.pth")
        print("Best Accuracy: ",best_val_accuracy,"%")

        # print the time elapsed
        end = time.time()
        elapsed = end-start
        minutes,seconds = divmod(elapsed,60)
        hours,minutes = divmod(minutes,60)
        print("Time Elapsed: {}h {}m {}s".format(int(hours),int(minutes),int(seconds)))

        print("Iterations:",iterations)
        print("Val_Accuracies:",val_accuracies)
        print("Val_Losses:",val_losses)
        print("Train_Losses:",train_losses)



    except KeyboardInterrupt:
        print("================================ QUIT Iteration {}================================\n Saving Model ...".format(i))
        torch.save(model.state_dict(),"../models/"+str(output_location)+"/MID_model_epoch"+str(epoch)+"_iter_"+str(i)+".pth")
        
        # validation pass
        accuracy,val_loss = eval_model(model,val_loader,device)
        val_accuracies.append(accuracy)
        val_losses.append(val_loss)

        # save the most accuracte model up to date
        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            torch.save(model.state_dict(),"../models/"+str(output_location)+"/BEST_model.pth")
        print("Best Accuracy: ",best_val_accuracy,"%")

        # print the time elapsed
        end = time.time()
        elapsed = end-start
        minutes,seconds = divmod(elapsed,60)
        hours,minutes = divmod(minutes,60)
        print("Time Elapsed: {}h {}m {}s".format(int(hours),int(minutes),int(seconds)))

        print("Iterations:",iterations)
        print("Val_Accuracies:",val_accuracies)
        print("Val_Losses:",val_losses)
        print("Train_Losses:",train_losses)


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
        print("\nValidation Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(eval_loss,correct,len(data_loader.dataset),100.*correct/len(data_loader.dataset)))

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

    print("Confusion Matrix")
    print(conf_mat)
    #print("Correct: ",preds.eq(labels).sum().item())


# --------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    train_SpeechPaceNN()
