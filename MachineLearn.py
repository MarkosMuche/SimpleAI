


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
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    return dataloader

def train_model(model,trainloader,classes):
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
    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    model.to(device)

    for ii, (inputs, labels) in enumerate(trainloader):
        # Move input and label tensors to the GPU
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ii==1:
            break
    return model
    print('hoola train is workign sof far')



