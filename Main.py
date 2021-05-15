
import torch
import threading
import MachineLearn
import os
import GUI
import helper
from torchvision import models
import cv2
import matplotlib.pyplot as plt
import csv
import PySimpleGUI as sg
import webbrowser
window =GUI.make_gui()
window.finalize()

toggle=True
# The following while loop waits until an event happens in window created above

while True:
    event, values =window.read()
    if event is None or event == 'Exit':
        break   
    # #########################################################if someone presses the training button on training tab
    if event=='Advanced settings':
        window['learning rate'].update(visible=toggle)
        window['lr'].update(visible=toggle)
        window['transfer learning model'].update(visible=toggle)
        window['transfer_models'].update(visible=toggle)
        window['epoch'].update(visible=toggle)
        window['ep'].update(visible=toggle)
        window['col1'].update(visible=toggle)
        window['col2'].update(visible=toggle)
        window['col3'].update(visible=toggle)
        window['col4'].update(visible=toggle)
        window['col5'].update(visible=toggle)
        window['col6'].update(visible=toggle)

    if event== 'train':
        transfer_model=values['transfer_models']
        learning_rate=float(values['lr'])
        epoch=int(values['ep'])

        DATADIR_train=values['train_folder']
        DATADIR_test=values['test_folder']
        model_name=values['model']
        
        if (len(DATADIR_train)==0 or len(DATADIR_test)==0 or len(model_name)==0):
            sg.popup('You have to input the training folder, the testing folder as well as the model!!!')
        else:
            classes=os.listdir(DATADIR_train)
            num_classes=len(classes)
            ####################################################################################################create model
            optimizer, model, input_size=helper.initialize_model(transfer_model,num_classes,learning_rate )
            dataloader_train, train_num_images=MachineLearn.prepare_data_train(DATADIR_train,input_size)
            dataloader_test, test_num_images=MachineLearn.prepare_data_test(DATADIR_test,input_size)

            window[GUI.train_out].update('number of training data = '+str(train_num_images),append=True)
            window[GUI.train_out].update('\nnumber of testing data = '+str(test_num_images),append=True)

            images,labels=next(iter(dataloader_train))
            
            ####################################################################################### saving to csv
            # saving the classes to csv
            csv_file=open(os.path.join('labels',model_name+'.csv'),'w+')
            with csv_file:
                write=csv.writer(csv_file)
                write.writerow(classes)

            # ###################################################################################the first thread for imshow
            t1=threading.Thread(target=helper.imageshow,args=(images[0],))
            t1.start()
            ################################################################################################another thread
            t2 = threading.Thread(target=MachineLearn.train_model, args=(optimizer, model,dataloader_train,dataloader_test, model_name, window,epoch), daemon=True)
            t2.start()
    
    ##############################################################if someone presses the predict button on the prediction tab
    if event=='predict':
        current_model_name= values['models_list']
        current_image_path=values['predict_image']
        
        if (len(current_model_name)==0 or len(current_image_path)==0):
            sg.popup('you have to select a model and an image!!')
        else:
            current_model_path=os.path.join('models',current_model_name+'.pth')
            current_model=torch.load(current_model_path)
            current_model.eval()
            current_image=helper.image_loader(current_image_path)
            logps=current_model(current_image)        
            probs=torch.exp(logps)
            top_p, top_class = probs.topk(1, dim=1)
            pred_index=top_class.item()
            probability=top_p.item()
            # read the classes csv
            class_labels=[]
            with open(os.path.join('labels',current_model_name +'.csv'),newline='' ) as cfile:
                rlist=csv.reader(cfile)
                for row in rlist:
                    class_labels.append(row)
            class_labels=class_labels[0]
            window[GUI.predict_out].update('\n this is '+class_labels[pred_index] + ' with probability of ' + str(probability*100)+'.', text_color_for_value='green',background_color_for_value='white',append=True)
    if event=='About...':
        sg.popup('Simple AI is a startup that develops an application for desktop. The application makes use of already available machine learning libraries like tensorflow and pytorch to make machine learning easy. AI is a field that is very applicable in the current world. However, making an implementable AI app takes an expert to go into the field and program it. This application makes it easy to implement any kind of AI algorithm easily. Using the app any person can train machine learning models easily without a deep knowledge of the field and no knowledge of programming. It is a very important app to implement AI solutions for companies. The application is on development stage and it will be released for trial soon.')

    if event=='documentation':
        webbrowser.open_new(r'docs\doc.pdf')

window.close()