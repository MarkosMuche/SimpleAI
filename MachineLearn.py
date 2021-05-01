
import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn
from torch import optim
import torch.nn.functional as F
import helper
# The following function uses a datasets.ImageFolder from torchvision to read image data from local folder and
# torch.utils.data.Dataloader class for converting our data into an iterable of batches that is shuffled
def prepare_data(data_dir):
    transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor()])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    num_data=len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    return dataloader, num_data

def train_model(model,trainloader,testloader,classes):
    for param in model.parameters():
        param.requires_grad = False
    from collections import OrderedDict
    num_classes=len(classes)
    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(1024, 500)),
                            ('relu', nn.ReLU()),
                            ('fc2', nn.Linear(500, num_classes)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))    
    model.classifier = classifier
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Only train the classifier parameters, feature parameters are frozen
    model.to(device)
    epochs = 10
    steps = 0
    running_loss = 0
    print_every = 2 
    for epoch in range(epochs):     # for each epoch 
        print('epoch', epoch, 'started...')
        running_loss, steps=forward_train(model, trainloader,device,running_loss,steps) # for each batch
        print('running loss in step', steps, ' is ',running_loss)    
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval() #turn to evaluation mode by making the dropouts to 0
            with torch.no_grad():
                test_loss, accuracy = forward_test(model, testloader, device,test_loss, accuracy)
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                f"Train loss: {running_loss/print_every:.3f}.. "
                f"Test loss: {test_loss/len(testloader):.3f}.. "
                f"Test accuracy: {accuracy/len(testloader):.3f}")
            running_loss = 0
            model.train()
        torch.save(model,'myModel.pth')
        #model = torch.load('checkpoint.pth')

    print('hoola your training has finished')
    return model
    

def forward_train(model, dataloader, device, running_loss, steps):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    for inputs, labels in dataloader:  # for each batch
        steps += 1 # which batch in the epoch
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs) # the output of nn.LogSoftmax(dim=1) from output layer
        loss = criterion(logps, labels)  # the loss between log probabilities and labels from criterion of NLLLoss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    # update all the weights (back propagation)
        running_loss += loss.item()
        # print('')
    return running_loss, steps

def forward_test(model, dataloader, device,test_loss, accuracy):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels) 
        test_loss += batch_loss.item()
        # Calculate accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    return test_loss, accuracy


def prepare_data_for_prediction():
    pass


def predict():
    pass
