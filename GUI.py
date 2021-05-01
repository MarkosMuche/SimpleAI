#!/usr/bin/env python
import PySimpleGUI as sg
def second_window():
    layout = [[sg.Text('The second form is small \nHere to show that opening a window using a window works')],
              [sg.OK()]] 
    window = sg.Window('Second Form', layout)
    event, values = window.read()
    window.close()
def gui_train():
    sg.theme('LightGreen')
    sg.set_options(element_padding=(0, 0))
    menu_def = [['&File', ['&Open', '&Save', '&Properties', 'E&xit' ]],
                ['&Edit', ['&Paste', ['Special', 'Normal',], 'Undo'],],
                ['&Toolbar', ['---', 'Command &1', 'Command &2', '---', 'Command &3', 'Command &4']],
                ['&Help', '&About...'],]
    right_click_menu = ['Unused', ['Right', '!&Click', '&Menu', 'E&xit', 'Properties']]
    # ------ GUI Defintion ------ #
    menu=[sg.Menu(menu_def, tearoff=False, pad=(20,1))]
    # create one tab for learning task
    layout_learn = [ menu,
              [sg.Text('What do you want to train?'), sg.InputText(key='model')],
              [sg.Text('Training folder'),sg.In(size=(25, 1), enable_events=True, key="train_folder"),sg.FolderBrowse()],
              [sg.Text('Testing folder'),sg.In(size=(25, 1), enable_events=True, key="test_folder"), sg.FolderBrowse()],
              [sg.Button('Submit',key='submit')],
              [sg.Output(size=(110, 30), font=('Helvetica 10'))]]
    # create another task for prediction
    layout_predict=[[sg.T('Here you can select from the models that you trained so far and predict')],
                [sg.Listbox('list of shit')],
                sg.Text('Training folder'),sg.In(size=(25, 1), enable_events=True, key="train_folder"),sg.FolderBrowse()]]
    tabs=[[sg.TabGroup([[sg.Tab('Training tab', layout_learn,title_color='Red',border_width=20,background_color='Green',
                        element_justification='center'), sg.Tab('Predict',layout_predict)]],tab_location='centertop',title_color='Red')]]
    window = sg.Window("Windows-like program",
                       tabs,
                       default_element_size=(12, 1),
                       grab_anywhere=False,
                       right_click_menu=right_click_menu,
                       default_button_element_size=(12, 1),
                       size=(400,400),
                       resizable=True)    
    return window