
#this python file is used to write some helper functions throughout, 
# i used udacity's pytorch code helper.py as a role model

import matplotlib.pyplot as plt
# The following function is used for showing images by matplotlib. 
# The images loaded by dataloader are shaped (3,224,224).
#  However, matplotlib plots images of shape(224,224,3). t
# his function uses torch.swapaxes() method to switch the axes to be visible by matplotlib.

def imageshow(image):
    image=image.swapaxes(0,2)
    image=image.swapaxes(0,1)
    plt.imshow(image)
    plt.show()

