#!/usr/bin/env python
import PySimpleGUI as sg
import os
####################################################################### read files from the models folder
models_list_pth=os.listdir('models')
models_list=[]

predict_out = 'predict_ml'+sg.WRITE_ONLY_KEY
train_out='train_ml'+sg.WRITE_ONLY_KEY

for model_list_pth in models_list_pth:
    models_list.append(model_list_pth.replace('.pth',''))

transfer_list=['resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception']

def make_gui():
    sg.theme('LightGreen')
    sg.set_options(element_padding=(0, 0))
    menu_def = [['&File', ['E&xit' ]],
                ['&Help', ['&About...', 'documentation']]]
    right_click_menu = ['Unused', ['Right', '!&Click', '&Menu', 'E&xit', 'Properties']]
    # ------ GUI Defintion ------ #
    menu=[sg.Menu(menu_def, tearoff=False, pad=(20,1))]
    # create one tab for learning task
    layout_learn = [ menu,
                    [sg.Combo(transfer_list,key='transfer_models',size=(10,1))],
                    [sg.Text('What do you want to train?'), sg.InputText(key='model')],
                    [sg.Text('Training folder'),sg.In(size=(25, 1), enable_events=True, key="train_folder"),sg.FolderBrowse()],
                    [sg.Text('Testing folder'),sg.In(size=(25, 1), enable_events=True, key="test_folder"), sg.FolderBrowse()],
                    [sg.Button('Start training',key='train',bind_return_key=True)],
                    [sg.Multiline(size=(200,30), key=train_out)],
                   ]
    # create another tab for prediction
    layout_predict=[[sg.T('Here you can select from the models that you trained so far and predict')],
                    [sg.Combo(models_list,key='models_list',size=(10,1))],
                    # the following is a file browser, above we used folder browsers
                    [sg.T("")], [sg.Text("Choose a file to predict: "), sg.Input(size=(25,1),tooltip='select image'), sg.FileBrowse(key='predict_image')],
                    [sg.Button('Predict', key='predict',bind_return_key=True)],
                    [sg.Multiline(size=(200,30), key=predict_out)],
                   ]

    ################################################################################# tab Group               
    tabs=[[sg.TabGroup([
            [sg.Tab('Training tab', layout_learn,title_color='Red',border_width=20,background_color='Cyan',element_justification ='center'), 
            sg.Tab('Prediction tab',layout_predict,element_justification ='center')]
            ],
            tab_location='centertop',title_color='Red')]]
    window = sg.Window("SimpleAI",
                       tabs,
                       default_element_size=(12, 1),
                       grab_anywhere=False,
                       right_click_menu=right_click_menu,
                       default_button_element_size=(12, 1),
                       size=(600,600),
                       resizable=True)    
    return window