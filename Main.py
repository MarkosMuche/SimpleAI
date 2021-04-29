
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
        dataloader_train=MachineLearn.prepare_data(DATADIR_train)
        dataloader_test=MachineLearn.prepare_data(DATADIR_test)
        images,labels=next(iter(dataloader_train))
        classes=os.listdir(DATADIR_train)
        t1=threading.Thread(target=helper.imageshow,args=(images[0],))
        t1.start()
        model = models.densenet121(pretrained=True)
        #another thread
        thread = threading.Thread(target=MachineLearn.train_model, args=(model,dataloader_train,classes), daemon=True)
        thread.start()
        # model=executor.submit( MachineLearn.train_model, args=[model,dataloader_train,classes], daemon=True).start()
        print('hello')

