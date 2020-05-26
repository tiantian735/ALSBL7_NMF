
#####################This is a self-designed class for import custom colormap from text file.########################
#####################################################################################################################
from os.path import dirname, abspath, basename
from os.path import join, exists
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt

class CustomColorMap:
    def __init__(self, cnumber):
    
        color = np.loadtxt(join(dirname(__file__),'all_ct.txt'),skiprows=1) #color profile
        colorname = np.loadtxt('colorname.csv',delimiter=',',dtype=str) #color number and number
        
        color_data = np.split(color,43)
        color_map = ListedColormap(color_data[cnumber]/np.amax(color_data[cnumber]),name=colorname)
        plt.register_cmap(name=colorname[cnumber][1],cmap=color_map)
        
    