#!/usr/bin/env python
from typing import Text
import PySimpleGUI as sg
import os
import helper
from PySimpleGUI.PySimpleGUI import Input, VSeparator
####################################################################### read files from the models folder
models_list_pth=os.listdir('models')
models_list=[]

predict_out = 'predict_ml'+sg.WRITE_ONLY_KEY ### keys for multilines
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
    menu=[sg.Menu(menu_def, tearoff=False)]
    # create one tab for learning task
    layout_learn_1=[
                    [sg.Text('Sample Images from the training folder')],
                    [sg.Image(key='train_image1',data=helper.get_img_data_tkinter('logos\SAI_logo1.jpg', first=True))],
                    [sg.Image(key='train_image2',data=helper.get_img_data_tkinter('logos\SAI_logo2.png', first=True))],
                    [sg.Image(key='train_image3',data=helper.get_img_data_tkinter('logos\SAI_logo1.jpg', first=True))]
    ]
    layout_learn_2 = [ menu,
                    [sg.Button('Advanced settings')],
                    [sg.Column([[sg.Text('learning rate',visible=False,key='learning rate')]],pad=(2,3),key='col6'),
                    sg.Column([[sg.Input('0.003',key='lr',size=(10,1),visible=False)]],pad=(2,3),key='col5')],
                    [sg.Column([[sg.Text('transfer learning model',key='transfer learning model',visible=False)]],
                    pad=(2,3),key='col1'),sg.Column([[sg.Combo(transfer_list,key='transfer_models',size=(10,1),
                    default_value='resnet',visible=False)]],pad=(2,3),key='col2'),sg.Column([[sg.Text('epoch',
                    key='epoch',visible=False)]],pad=(2,3),key='col3'),sg.Column([[sg.Input('5',key='ep',visible=False)]],
                    pad=(2,3),key='col4')],
                    [sg.Text('What do you want to train?'), sg.InputText(key='model')],
                    [sg.Text('Training folder'),sg.In(size=(25, 1), enable_events=True, key="train_folder"),sg.FolderBrowse()],
                    [sg.Text('Testing folder'),sg.In(size=(25, 1), enable_events=True, key="test_folder"), sg.FolderBrowse()],
                    [sg.Button('Start training',key='train',bind_return_key=True)],
                    [sg.Multiline(size=(250,24), key=train_out)]
                   ]

    layout_learn=[[
                sg.Column(layout_learn_1),
                sg.VSeparator(),
                sg.Column(layout_learn_2)

    ]]
    # create another tab for prediction
    layout_predict_1=[ 
        [sg.Text(" Predictiona Images List"),sg.In(size=(25, 1), enable_events=True, key="predict_folder"),sg.FolderBrowse(),],
        [sg.Listbox(values=[], enable_events=True, size=(60, 60), key="images_list")],]


    layout_predict_2=[[sg.T('Select from the models that you trained and predict', pad=(5,30), font='Courier')],
                    [sg.T('Choose your model'),sg.Combo(models_list,key='models_list',size=(10,1))],
                    [sg.Text("Choose a file to predict: "), sg.Input(size=(25,1),tooltip='select image'), sg.FileBrowse(key='predict_image')],
                    [sg.Button('Predict', key='predict',bind_return_key=True)],
                    [sg.Image(key='image',size=(80,70))],
                    [sg.Multiline(size=(200,40), key=predict_out)]
                   ]

    layout_predict=[[
        sg.Column(layout_predict_1),
        sg.VSeparator(),
        sg.Column(layout_predict_2)
    ]]
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
                       resizable=True)    
    return window