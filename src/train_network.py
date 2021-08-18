'''
network_train.py

This file builds the dataset, dataloader, NN,
and defines the training loop and stat helper functions
'''

from torch.optim import optimizer
from pytorch_dataset import SpeechPaceDataset,my_collate_fn,FusedDataset, my_collate_fn_fused
from network_def import SpeechPaceNN,PMRfusionNN
import torch
from torchaudio import transforms,utils
from torch.utils.data import Dataset, DataLoader
import time
import sys
from torchsampler import ImbalancedDatasetSampler
import argparse
import os

# ===================================== Training Function =====================================
def train_nn(args):
    # get the device, hopefully a GPU
    torch.cuda.set_device(int(args.gpu_i))
    device = torch.device("cuda:"+str(args.gpu_i) if torch.cuda.is_available() else "cpu")

    # print model training info
    print("\n\n\n================================ Start Training ================================")
    print("\nSession Name:",args.session_name)
    print("\nModel Name:",args.model_name)
    print("\nDevice:",torch.cuda.current_device()," ----> ",torch.cuda.get_device_name(torch.cuda.current_device()))
    print("\nHyperparameters:")

    print("Batch Size: {}\nLearning Rate: {}\nHidden Size: {}\nNumber of Layer: {}\nNumber of Epochs: {}\nNormalization:{}".format(\
        args.batch_size,args.lr,args.hidden_size,args.num_layers,args.num_epochs,args.normalize))

    # global variables
    best_val_accuracy = 0
    train_losses = []
    val_losses = []
    val_accuracies = []
    iterations = []
    curr_train_loss = 0

    # Create and load the datasets
    train_dataset = create_dataset(args,args.train_labels_csv,args.train_data_dir)
    val_dataset = create_dataset(args,args.val_labels_csv,args.val_data_dir)
    #test_dataset = create_dataset(args,test_labels_csv,args.test_data_dir)

    train_loader = create_loader(train_dataset,args)
    val_loader = create_loader(val_dataset,args)
    #test_loader = create_loader(test_dataset,args)
    
    # build and load the model
    model,criterion = create_model(args)
    model.to(device)

    # optimization criteria
    optimizer = create_optimizer(model,args)

    try:
        # track model training time
        start = time.time()

        # track total iterations
        num_iter = 0

        # ----------------------------------------------------- Training Loop -----------------------------------------------------
        for epoch in range(args.num_epochs):
            # get the next batch
            for i, batch in enumerate(train_loader):

                # load the data and labels to the gpu
                data,labels = to_gpu(batch,device,args)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward pass
                out = forward_pass(data,batch,model,args)

                # backward pass
                loss = criterion(out,labels)
                if torch.isnan(loss):
                    print("********** NAN ERROR ************")
                    print("output:",out)
                    print("labels:",labels)
                    print("idxs:",get_idxs(batch))
                    exit()
                curr_train_loss += loss.item()
                loss.backward()
                optimizer.step()
                
                # print training statistics every n batches
                if i % args.loss_freq == 0 and i != 0:
                    print("Train Epoch: {} Iteration: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}".format(epoch,i,i*args.batch_size,len(train_loader.dataset),100.*i/len(train_loader),loss.item()))
                
                # do a validation pass every m batches (may not want to wait till the end of an epoch)
                if i % args.val_freq == 0 and i != 0:
                    print("\n\n----------------- Epoch {} Iteration {} -----------------\n".format(epoch,i))

                    # keep track of training and validation loss, since training forward pass takes a while do every m iterations instead of every epoch
                    num_iter += args.val_freq
                    iterations.append(num_iter)
                    train_losses.append(curr_train_loss/(args.val_freq)) # average batch training loss over m iterations instead of over the entire dataset
                    curr_train_loss = 0 # reset

                    # validation pass
                    accuracy,val_loss = eval_model(model,val_loader,device,args)
                    val_accuracies.append(accuracy)
                    val_losses.append(val_loss)

                    # save the most accuracte model up to date
                    if accuracy > best_val_accuracy:
                        best_val_accuracy = accuracy
                        torch.save(model.state_dict(),args.log_dest+"BEST_model.pth")
                    print("Best Accuracy: ",best_val_accuracy,"%")

                    # print the time elapsed
                    end = time.time()
                    elapsed = end-start
                    minutes,seconds = divmod(elapsed,60)
                    hours,minutes = divmod(minutes,60)
                    print("Time Elapsed: {}h {}m {}s".format(int(hours),int(minutes),int(seconds)))
                    print("\n--------------------------------------------------------\n\n")

        print("================================ Finished Training ================================")
        torch.save(model.state_dict(),args.log_dest+"END_model.pth")
        
        # validation pass
        accuracy,val_loss = eval_model(model,val_loader,device,args,print_idxs=True)
        
        # save the most accuracte model up to date
        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            torch.save(model.state_dict(),args.log_dest+"BEST_model.pth")
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
        torch.save(model.state_dict(),args.log_dest+"MID_model.pth")
        print("================================ QUIT ================================\n Saving Model ...") 
        
        # validation pass
        accuracy,val_loss = eval_model(model,val_loader,device)

        # save the most accuracte model up to date
        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            torch.save(model.state_dict(),args.log_dest+"BEST_model.pth")
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

    
# ===================================== Model Dependent Functions =====================================

# create the dataset used by the corresponding model
def create_dataset(args,labels_csv,data_dir):
    if args.model_name == "SpeechPaceNN":
        return SpeechPaceDataset(os.path.join(args.root_dir,data_dir),os.path.join(args.root_dir,data_dir,labels_csv))
    elif args.model_name == "PMRfusionModel":
        return FusedDataset(args.root_dir,os.path.join(args.root_dir,labels_csv))
    else:
        print("ERROR: invalid model name")
        exit(1)

# create the dataloader based on the collate_fn used by the corresponding model
def create_loader(dataset,args):
    if args.model_name == "SpeechPaceNN":
        return DataLoader(dataset,args.batch_size,collate_fn=my_collate_fn)
    elif args.model_name == "PMRfusionModel":
        return DataLoader(dataset,args.batch_size,collate_fn=my_collate_fn_fused)
    else:
        print("ERROR: invalid model name")
        exit(1)

# create the optimizer specified
def create_optimizer(model,args):
    if args.optim == "Adam":
        return torch.optim.Adam(model.parameters(),lr=args.lr)
    elif args.optim == "SGD":
        return torch.optim.SGD(model.parameters(),lr=args.lr)
    else:
        print("ERROR: invalid optimizer name")
        exit(1)

# create the model and loss function based on the name specified and classification/regression options
def create_model(args):
    if args.model_name == "SpeechPaceNN":
        if args.classification == "y":
            return SpeechPaceNN(args.input_size,args.hidden_size,args.num_layers,args.num_classes,args.gpu_i,args.hidden_init_rand),torch.nn.CrossEntropyLoss()
        elif args.regression == "y":
            return SpeechPaceNN(args.input_size,args.hidden_size,args.num_layers,-1,args.gpu_i,args.hidden_init_rand),torch.nn.MSELoss()
    elif args.model_name == "PMRfusionModel":
        if args.classification == "y":
            return PMRfusionNN(args.input_size,args.hidden_size,args.num_layers,args.num_classes,args.gpu_i,args.hidden_init_rand),torch.nn.CrossEntropyLoss()
        elif args.regression == "y":
            return PMRfusionNN(args.input_size,args.hidden_size,args.num_layers,-1,args.gpu_i,args.hidden_init_rand),torch.nn.MSELoss()
    else:
        print("ERROR: invalid model name")
        exit(1)

# load the batch to the gpu, based on the model the batch has a different structure
# return as (data),labels
def to_gpu(batch,device,args):
    if args.model_name == "SpeechPaceNN":
        X_audio,labels = batch[0].to(device),batch[2].to(device)
        return X_audio,labels
    elif args.model_name == "PMRfusionModel":
        X_audio,X_video,labels = batch[0].to(device),batch[1].to(device),batch[4].device
        return (X_audio,X_video),labels
    else:
        print("ERROR: invalid model name")
        exit(1)

# do a forward pass using the data and batch structure for the corresponding model
def forward_pass(data,batch,model,args):
    if args.model_name == "SpeechPaceNN":
        # X_audio = data
        # lengths_audio = batch[1]
        return model(data,batch[1])
    elif args.model_name == "PMRfusionModel":
        X_audio,X_video = data[0],data[1]
        # lengths_audio, lengths_video = batch[2],batch[3]
        return model(X_audio,batch[2],X_video,batch[3])
    else:
        print("ERROR: invalid model name")
        exit(1)

# get the idxs based on the batch structure of the corresponding model
def get_idxs(batch,args):
    if args.model_name == "SpeechPaceNN":
        return batch[3]
    elif args.model_name == "PMRfusionModel":
        return batch[5]
    else:
        print("ERROR: invalid model name")
        exit(1)

'''Helper function to evaluate the network (used during training, validation, and testing)'''
def eval_model(model,data_loader,device,criterion,args,print_idxs=False):
    model.eval()
    model.to(device)

    eval_loss = 0
    correct = 0

    all_preds = torch.tensor([])
    all_labels = torch.tensor([],dtype=torch.int)
    all_idxs = torch.tensor([],dtype=torch.int)
    all_preds = all_preds.to(device)
    all_labels = all_labels.to(device)
    all_idxs = all_idxs.to(device)
    with torch.no_grad():
        val_start=time.time()
        for i, (batch) in enumerate(data_loader):
            data,labels = to_gpu(batch,device,args)
            idxs = get_idxs(batch,args).to(device)
            
            # forward pass
            out = forward_pass(data,batch,model,args)

            if args.classification:
                # accumulate predictions and labels
                all_preds = torch.cat((all_preds,out),dim=0)
                all_labels = torch.cat((all_labels,labels),dim=0)
                all_idxs = torch.cat((all_idxs,idxs),dim=0)

            # sum up the batch loss
            loss = criterion(out,labels)
            eval_loss += loss.item()

            if args.classification:
                # get the prediction
                pred = out.max(1,keepdim=True)[1]
                correct += pred.eq(labels.view_as(pred)).sum().item()

        print("validation computation time:",(time.time()-val_start)//60," minutes")
        if args.classification:
            gen_conf_mat(all_preds,all_labels,all_idxs,args.num_classes,print_idxs)
        num_batches = len(data_loader.dataset)//data_loader.batch_size
        eval_loss /= num_batches
        print("\nValidation Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(eval_loss,correct,len(data_loader.dataset),100.*correct/len(data_loader.dataset)))

    model.train()
    return 100.*correct/len(data_loader.dataset),eval_loss


'''Helper function to create a confusion matrix of classification results'''
# this gets called by eval_model with the predictions and labels
def gen_conf_mat(predictions,labels,idxs,num_classes,print_idxs=False):
    # get the prediction from the max output
    preds = predictions.argmax(dim=1)

    # generate label-prediction pairs
    stacked = torch.stack((preds,labels),dim=1)

    # create the confusion matrix
    conf_mat = torch.zeros(num_classes,num_classes,dtype=torch.int64)

    if print_idxs == True:
        incorrect = []
        # fill the confusion matrix
        for i,pair in enumerate(stacked):
            x,y = pair.tolist()
            conf_mat[x,y] = conf_mat[x,y]+1
            if x!=y:
                incorrect.append(idxs[i].item())
        print("Incorrect Samples:",incorrect)
    else:
        # fill the confusion matrix
        for pair in stacked:
            x,y = pair.tolist()
            conf_mat[x,y] = conf_mat[x,y]+1

    print("Confusion Matrix")
    print(conf_mat)

# ===================================== Main =====================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training and Evaluation")
    # logging details
    parser.add_argument("session_name",help="prefix the logging directory with this name",type=str)
    #parser.add_argument("log_dest",help="name of directory with logging info (stats, train model, parameters, etc.)",type=str)

    # dataset details
    parser.add_argument("root_dir",help="full path to dataset directory",type=str)
    parser.add_argument("--train_data_dir",help="specify if training data is in another directory within root_dir",type=str)
    parser.add_argument("--val_data_dir",help="specify if validation data is in another directory within root_dir",type=str)
    #parser.add_argument("--test_data_dir",help="specify if test data is in another directory within root_dir",type=str)
    parser.add_argument("train_labels_csv",help="file name of csv with training labels and/or metadata",type=str)
    parser.add_argument("val_labels_csv",help="file name of csv with validation labels and/or metadata",type=str)
    # parser.add_argument("test_labels_csv",help="file name of csv with test labels and/or metadata",type=str)

    # training details
    parser.add_argument("gpu_i",help="GPU instance to use (0, 1, 2, etc.)",type=int)
    parser.add_argument("model_name",help="name of the model class (torch.nn.Module)",type=str)
    parser.add_argument("optim",help="name of PyTorch optimizer to use",type=str)
    parser.add_argument("loss_freq",help="print the loss every nth batch",type=int)
    parser.add_argument("val_freq",help="do a validation pass every nth batch",type=int)

    # hyperparameters
    parser.add_argument("batch_size",help="what size batch to use (32, 64, 128, etc.)",type=int)
    parser.add_argument("lr",help="learning rate",type=float)
    parser.add_argument("hidden_size",help="dimension of hidden state (RNN)",type=int)
    parser.add_argument("classification",help="use the model for classification (y/n)",type=str)
    parser.add_argument("num_classes",help="number of classes for the task (set to -1 for regression)",type=int)
    parser.add_argument("regression",help="use the model for regression (y/n)",type=str)
    parser.add_argument("input_size",help="dimension of input features (e.g. MFCC features = 26)",type=int)
    parser.add_argument("num_layers",help="number of GRU layers",type=int)
    parser.add_argument("num_epochs",help="number of times to go through entire training set",type=int)
    parser.add_argument("normalize",help="normalize input features (y/n)",type=str)
    parser.add_argument("hidden_init_rand",help="initialize the hidden state with random values, otherwise use zeros (y/n)",type=str)

    args = parser.parse_args()

    if args.classification == "y" and args.regression == "y":
        parser.error("can only choose either classification or regression, not both")
    if args.classification == "y" and args.num_classes <= 0:
        parser.error("chose classification but --num_classes is invalid, must specify --num_classes")
    if args.regression == "y" and args.num_classes > 0:
        parser.error("chose regression but also specified --num_classes, invalid selection")
    
    train_nn(args)
