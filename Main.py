
import torch
import threading
import MachineLearn
#import TrainModel
import os
import GUI
import helper
from torchvision import models

window =GUI.gui_train()

# The following while loop waits until an event happens in window created above
while True:
    event, values =window.read()
    if event is None or event == 'Exit':
        break
        print('Event = ', event)
    if event== 'submit':
        DATADIR_train=values['train_folder']
        DATADIR_test=values['test_folder']
        model_name=values['model'] 
               
        dataloader_train, num_data_train=MachineLearn.prepare_data(DATADIR_train)
        dataloader_test, num_data_test=MachineLearn.prepare_data(DATADIR_test)
        print('number of training data = ', num_data_train)
        print('number of testing data = ', num_data_test)

        images,labels=next(iter(dataloader_train))
        classes=os.listdir(DATADIR_train)
        # the first thread for imshow
        t1=threading.Thread(target=helper.imageshow,args=(images[0],))
        t1.start()
        model = models.densenet121(pretrained=True)
        #another thread
        t2 = threading.Thread(target=MachineLearn.train_model, args=(model,dataloader_train,dataloader_test, classes), daemon=True)
        t2.start()