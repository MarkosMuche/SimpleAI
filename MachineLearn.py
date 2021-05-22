

import os
import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision.transforms.transforms import Resize
import helper
import GUI
#################################################### global variables
# The following function uses a datasets.ImageFolder from torchvision to read image data from local folder and
# torch.utils.data.Dataloader class for converting our data into an iterable of batches that is shuffled
def prepare_data_train(data_dir, input_size):
    train_transforms = transforms.Compose([transforms.Resize(300),
                                        transforms.CenterCrop(input_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    dataset = datasets.ImageFolder(data_dir, transform=train_transforms)
    num_images=len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    return dataloader, num_images


def prepare_data_test(data_dir,input_size):
    test_transforms = transforms.Compose([transforms.Resize(300),
                                      transforms.CenterCrop(input_size),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    dataset = datasets.ImageFolder(data_dir, transform=test_transforms)
    num_images=len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    return dataloader, num_images


def train_model(optimizer, model,trainloader,testloader,model_name, window, epochs ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Only train the classifier parameters, feature parameters are frozen
    model.to(device)
    steps = 0
    running_loss = 0
    print_every = 1
    for epoch in range(epochs):     # for each epoch 
        window[GUI.train_out].update('\nepoch '+str(epoch+1) + ' started...',append=True)
        running_loss, steps=forward_batch(optimizer, window, model, trainloader,device,running_loss,steps) 
        running_loss,steps=round(running_loss,4),round(steps,4)
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval() #turn to evaluation mode by making the dropouts to 0
            with torch.no_grad():
                test_loss, accuracy = forward_test(model, testloader, device,test_loss, accuracy)
                test_loss, accuracy=round(test_loss,4),round(accuracy,4)
                    
            window[GUI.train_out].update('\nEpoch' + str(epoch+1)+'/'+str(epochs)+'..'+
                'Train loss: '+str(running_loss/print_every)+'.. '+
                'Test loss: '+str(test_loss/len(testloader))+'.. '+
                'Test accuracy: '+str(accuracy),append=True)
            running_loss = 0
            model.train()
        path_save=os.path.join('models',model_name+'.pth')
        torch.save(model,path_save)
    window[GUI.train_out].update('\nhoola model has finished training.',append=True)
    

def forward_batch(optimizer, window, model, dataloader, device, running_loss, steps):
    criterion = nn.NLLLoss()
    for inputs, labels in dataloader:  # for each batch
        steps += 1 # which batch in the epoch
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model(inputs) # the output of nn.LogSoftmax(dim=1) from output layer
        loss = criterion(logps, labels)  # the loss between log probabilities and labels from criterion of NLLLoss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    # update all the weights (back propagation)
        running_loss += loss.item()
        window[GUI.train_out].update('>',append=True)
    return running_loss, steps

def forward_test(model, dataloader, device,test_loss, accuracy):
    criterion = nn.NLLLoss()
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model(inputs)
        batch_loss = criterion(logps, labels) 
        test_loss += batch_loss.item()
        ########################################################################################## Calculate accuracy
        ps = torch.exp(logps)
        _, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    return test_loss, accuracy/(len(dataloader))