
#this python file is used to write some helper functions throughout, 
# i used udacity's pytorch code helper.py as a role model

import matplotlib.pyplot as plt
from torch import nn, optim
from torchvision import models
import os
import cv2
import torch
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms.transforms import CenterCrop
# The following function is used for showing images by matplotlib. 
# The images loaded by dataloader are shaped (3,224,224).
#  However, matplotlib plots images of shape(224,224,3). t
# his function uses torch.swapaxes() method to switch the axes to be visible by matplotlib.
learning_rate=0.003
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def imageshow(image):
    image=image.swapaxes(0,2)
    image=image.swapaxes(0,1)
    plt.imshow(image)
    plt.show()

imsize = 224
loader = transforms.Compose([transforms.Resize(255),transforms.CenterCrop(imsize), transforms.ToTensor()])

def image_loader(image_path):
    """load image, returns cuda tensor"""
    image = Image.open(image_path)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image.to(device)


def initialize_model(model_name, num_classes):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=True)
        for param in model_ft.parameters():
            param.requires_grad = False
        num_ftrs = model_ft.fc.in_features
        from collections import OrderedDict
        model_ft.fc = nn.Sequential(OrderedDict([
            ('fc1',nn.Linear(num_ftrs,500)),
            ('relu',nn.ReLU()),
            ('fc2',nn.Linear(500,num_classes)),
            ('output',nn.LogSoftmax(dim=1))
        ]))
        print(model_ft.fc)
        print(model_ft.fc.requires_grad_)
        input_size = 224
        optimizer = optim.Adam(model_ft.fc.parameters(), lr=learning_rate)
        

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=True)
        for param in model_ft.parameters():
            param.requires_grad = False
        num_ftrs = model_ft.classifier[6].in_features
        from collections import OrderedDict
        model_ft.classifier[6] = nn.Sequential(OrderedDict([
            ('fc1',nn.Linear(num_ftrs,500)),
            ('relu',nn.ReLU()),
            ('fc2',nn.Linear(500,num_classes)),
            ('output',nn.LogSoftmax(dim=1))
        ]))
        input_size = 224
        optimizer = optim.Adam(model_ft.classifier[6].parameters(), lr=learning_rate)

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=True)
        for param in model_ft.parameters():
            param.requires_grad = False
        num_ftrs = model_ft.classifier[6].in_features
        from collections import OrderedDict
        model_ft.classifier[6] = nn.Sequential(OrderedDict([
            ('fc1',nn.Linear(num_ftrs,500)),
            ('relu',nn.ReLU()),
            ('fc2',nn.Linear(500,num_classes)),
            ('output',nn.LogSoftmax(dim=1))
        ]))
        input_size = 224
        optimizer = optim.Adam(model_ft.classifier[6].parameters(), lr=learning_rate)

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=True)
        for param in model_ft.parameters():
            param.requires_grad = False
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        num_ftrs = model_ft.classifier[1].in_features
        from collections import OrderedDict
        model_ft.classifier[1] = nn.Sequential(OrderedDict([
            ('fc1',nn.Linear(num_ftrs,500)),
            ('relu',nn.ReLU()),
            ('fc2',nn.Linear(500,num_classes)),
            ('output',nn.LogSoftmax(dim=1))
        ]))
        model_ft.num_classes=num_classes
        input_size = 224
        optimizer = optim.Adam(model_ft.classifier[1].parameters(), lr=learning_rate)

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=True)
        for param in model_ft.parameters():
            param.requires_grad = False
        num_ftrs = model_ft.classifier.in_features

        from collections import OrderedDict
        model_ft.classifier = nn.Sequential(OrderedDict([
            ('fc1',nn.Linear(num_ftrs,500)),
            ('relu',nn.ReLU()),
            ('fc2',nn.Linear(500,num_classes)),
            ('output',nn.LogSoftmax(dim=1))
        ]))        
        input_size = 224
        optimizer = optim.Adam(model_ft.classifier.parameters(), lr=learning_rate)

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=True)
        for param in model_ft.parameters():
            param.requires_grad = False
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        from collections import OrderedDict
        model_ft.fc = nn.Sequential(OrderedDict([
            ('fc1',nn.Linear(num_ftrs,500)),
            ('relu',nn.ReLU()),
            ('fc2',nn.Linear(500,num_classes)),
            ('output',nn.LogSoftmax(dim=1))
        ]))
        input_size = 299
        optimizer = optim.Adam(model_ft.fc.parameters(), lr=learning_rate)

    else:
        print("Invalid model name, exiting...")
        exit()

    return optimizer, model_ft, input_size


