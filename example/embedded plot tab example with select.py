import wx
import wx.lib.agw.aui as aui
import wx.lib.mixins.inspection as wit
import matplotlib.cm as cm

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar

from os.path import join, exists
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector





def demo():
    def onselect(eclick, erelease):
        global coordinate_new
        print('startposition: (%f, %f)' % (eclick.xdata, eclick.ydata))
        print('endposition  : (%f, %f)' % (erelease.xdata, erelease.ydata))
        print('used button  : ', eclick.button)
        coordinate_new=toggle_selector.RS.geometry
        coordinate_new=coordinate_new.astype(int)
        print(coordinate_new)
    
    

    def toggle_selector(event):
        global coorinate_new
        print('Key pressed.')
        if event.key in ['Q', 'q'] and toggle_selector.RS.active:
            print('RectangleSelector deactivated.')
            toggle_selector.RS.set_active(False)
        if event.key in ['A', 'a'] and not toggle_selector.RS.active:
            print('RectangleSelector activated.')
            toggle_selector.RS.set_active(True)
        if event.key in ['D', 'd'] and toggle_selector.RS.active:
            save_image(coordinate_new)
    # alternatively you could use
    #app = wx.App()
    # InspectableApp is a great debug tool, see:
    # http://wiki.wxpython.org/Widget%20Inspection%20Tool
    data = np.load('D:\\test1.npy')
    app = wit.InspectableApp()
    frame = wx.Frame(None, -1, 'Plotter')
    plotter = PlotNotebook(frame)
    
    axes1 = plotter.add(number=1,name='figure 1').gca()
    img1=axes1.imshow(data[1])
    Plot.figure.colorbar(img1, ax=axes1)
    #axes1.colorbar.ColorbarBase(ax=axes1, cmap=cm.Greys, 
                                # orientation="vertical")
    plt.subplot(1,2,2)
    plt.imshow(data[4000])
    plt.colorbar()
    axes1 = plotter.add(number=2,name='figure 1').gca()
    axes1.imshow(data[4000])
    
    #plt.subplot(1,2,2)
    #plt.plot([1, 2, 3, 4, 5], [2, 1, 4, 2, 3])
    
    toggle_selector_RS = RectangleSelector(axes1, onselect, drawtype='box',
                                            #minspanx=5, minspany=5, 
                                            spancoords='data', interactive=True, 
                                            )
    Plot.figure.canvas.mpl_connect('key_press_event', toggle_selector)
        
    
            
    rectprops = dict(facecolor='red', edgecolor = 'black',
                     alpha=0.2, fill=True)
                     
    
    #plt.subplot(1,2,1)
    #plt.plot([1, 2, 3], [2, 1, 4])
    #Plot.fig2.plot([1, 2, 3, 4, 5], [2, 1, 4, 2, 3])
    

    frame.Show()
    app.MainLoop()
    

class Plot(wx.Panel):
    def __init__(self, parent, number, id=-1, dpi=None,**kwargs):
        wx.Panel.__init__(self, parent,  id=id,**kwargs)
        Plot.figure,Plot.axes = plt.subplots(dpi=dpi, figsize=(2, 2))
        plt.subplot(1,2,1)
        
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.toolbar = NavigationToolbar(self.canvas)
        self.toolbar.Realize()
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas, 1, wx.EXPAND)
        sizer.Add(self.toolbar, 0, wx.LEFT | wx.EXPAND)
        self.SetSizer(sizer)
        
        
    def onselect(self,eclick, erelease):
        global coordinate_new
        print('startposition: (%f, %f)' % (eclick.xdata, eclick.ydata))
        print('endposition  : (%f, %f)' % (erelease.xdata, erelease.ydata))
        print('used button  : ', eclick.button)
        coordinate_new=toggle_selector.RS.geometry
        coordinate_new=coordinate_new.astype(int)
        print(coordinate_new)
    
    

    def toggle_selector(self,event):
        global coorinate_new
        print('Key pressed.')
        if event.key in ['Q', 'q'] and toggle_selector.RS.active:
            print('RectangleSelector deactivated.')
            toggle_selector.RS.set_active(False)
        if event.key in ['A', 'a'] and not toggle_selector.RS.active:
            print('RectangleSelector activated.')
            toggle_selector.RS.set_active(True)
        if event.key in ['D', 'd'] and toggle_selector.RS.active:
            save_image(coordinate_new)
            
        rectprops = dict(facecolor='red', edgecolor = 'black',
                     alpha=0.2, fill=True)



class PlotNotebook(wx.Panel):
    def __init__(self, parent, id=-1):
        wx.Panel.__init__(self, parent, id=id)
        self.nb = aui.AuiNotebook(self)
        sizer = wx.BoxSizer()
        sizer.Add(self.nb, 1, wx.EXPAND)
        self.SetSizer(sizer)

    def add(self,number,name="plot"):
        page = Plot(self.nb,number=number)
        
        self.nb.AddPage(page, name)
        return page.figure
        
        
if __name__ == "__main__":
    demo()