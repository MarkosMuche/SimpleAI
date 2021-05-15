#!/usr/bin/env python
from typing import Text
import PySimpleGUI as sg
import os

from PySimpleGUI.PySimpleGUI import Input
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
    menu_def = [['&File', ['E&xit' ]],
                ['&Help', ['&About...', 'documentation']]]
    right_click_menu = ['Unused', ['Right', '!&Click', '&Menu', 'E&xit', 'Properties']]
    # ------ GUI Defintion ------ #
    menu=[sg.Menu(menu_def, tearoff=False, pad=(20,1))]
    # create one tab for learning task
    layout_learn = [ menu,
                    [sg.Button('Advanced settings')],
                    [sg.Column([[sg.Text('learning rate',visible=False,key='learning rate')]],pad=(2,3),key='col6'),sg.Column([[sg.Input('0.003',key='lr',size=(10,1),visible=False)]],pad=(2,3),key='col5')],
                    [sg.Column([[sg.Text('transfer learning model',key='transfer learning model',visible=False)]],pad=(2,3),key='col1'),sg.Column([[sg.Combo(transfer_list,key='transfer_models',size=(10,1),default_value='resnet',visible=False)]],pad=(2,3),key='col2'),sg.Column([[sg.Text('epoch',key='epoch',visible=False)]],pad=(2,3),key='col3'),sg.Column([[sg.Input('5',key='ep',visible=False)]],pad=(2,3),key='col4')],
                    [sg.Text('What do you want to train?'), sg.InputText(key='model')],
                    [sg.Text('Training folder'),sg.In(size=(25, 1), enable_events=True, key="train_folder"),sg.FolderBrowse()],
                    [sg.Text('Testing folder'),sg.In(size=(25, 1), enable_events=True, key="test_folder"), sg.FolderBrowse()],
                    [sg.Button('Start training',key='train',bind_return_key=True)],
                    [sg.Multiline(size=(200,30), key=train_out)],
                   ]
    # create another tab for prediction
    layout_predict=[[sg.T('Here you can select from the models that you trained so far and predict')],
                    [sg.T('Choose your model'),sg.Combo(models_list,key='models_list',size=(10,1))],
                    [sg.Text("Choose a file to predict: "), sg.Input(size=(25,1),tooltip='select image'), sg.FileBrowse(key='predict_image')],
                    [sg.Button('Predict', key='predict',bind_return_key=True)],
                    [sg.Multiline(size=(200,30), key=predict_out)],
                   ]

    ################################################################################# tab Group               
    tabs=[[sg.TabGroup([
            [sg.Tab('Training tab', layout_learn,title_color='Black',border_width=20,background_color='Cyan',element_justification ='left'), 
            sg.Tab('Prediction tab',layout_predict,background_color='Cyan',element_justification ='left')]
            ],
            tab_location=['topleft','topright'],title_color='Black',selected_background_color='Cyan',tab_background_color='Grey')]]
    window = sg.Window("SimpleAI",
                       tabs,
                       default_element_size=(12, 1),
                       grab_anywhere=False,
                       right_click_menu=right_click_menu,
                       default_button_element_size=(12, 1),
                       size=(600,600),
                       resizable=True)    
    return window