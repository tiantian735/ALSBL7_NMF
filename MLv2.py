'''
Include Package
'''

from os.path import dirname, abspath, basename
from os.path import join, exists
from os import mkdir
from os import rmdir
from astropy.io import fits
from shutil import copy2
from glob import iglob
from datetime import datetime
import sys


import wx
import wx.lib.intctrl
from wx.lib.splitter import MultiSplitterWindow
from MyNumberValidator import MyNumberValidator

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFilter
from PIL import ImageChops
from CustomColorMap import CustomColorMap as ccm
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar
from matplotlib.widgets import RectangleSelector
import wx.lib.agw.aui as aui
import wx.lib.mixins.inspection as wit



import nimfa
from sklearn.decomposition import NMF


#create a temp file for images
path=join(dirname(__file__),"temp")
if not exists(path):
    mkdir(path)



    

    



class MainWindow(wx.Frame):

    def __init__(self, parent, title):
    
        wx.Frame.__init__(self, parent, title=title)
        self.SetMinSize((800, 600))
        self.Centre()
        
        
        #main= NMFAuiWindow(parent=self,name='nmf')
        #main.SetTopWindow(main)
        '''
        main=wx.Panel(self)
        mainsplitter = MultiSplitterWindow(self,id=-1, style=wx.SP_LIVE_UPDATE, name="multiSplitter")
        

        panelone=wx.Panel(main)
        nmfpanel(parent=panelone, nmf=nimfa)

        
        self.paneltwo=wx.Panel(self,size=(800,800))
        plotter = PlotNotebook(parent=self.paneltwo)
        axes1 = plotter.add('figure 1').gca()
        axes1.plot([1, 2, 3], [2, 1, 4])
        axes2 = plotter.add('figure 2').gca()
        axes2.plot([1, 2, 3, 4, 5], [2, 1, 4, 2, 3])
        self.Show()

        
        
        

        
        
        
        
        
        
        mainsplitter.AppendWindow(panelone)
        mainsplitter.AppendWindow(paneltwo)
        
        text = wx.TextCtrl(self, style= wx.TE_MULTILINE | wx.TE_READONLY)
        text.SetDefaultStyle(wx.TextAttr(wx.RED))
        text.AppendText("Red text\n")
        #text.SetDefaultStyle(wx.TextAttr(wx.NullColour, wx.LIGHT_GREY))
        #text.AppendText("Red on grey text\n")
        #text.SetDefaultStyle(wx.TextAttr(wx.BLUE))
        #text.AppendText("Blue on grey text\n")
        '''
        
        
        
        #self.CreateStatusBar() # A Statusbar in the bottom of the window
        
        
        #Setup file menu.
        filemenu=wx.Menu()
        
        menuOpen=filemenu.Append(wx.ID_OPEN, "&Open"," Open a FITS or NPY file")
        filemenu.AppendSeparator()
        menuSave=filemenu.Append(wx.ID_SAVE, "&Save"," Save opened file into NPY form")
        menuExport=filemenu.Append(wx.ID_ANY, "&Export"," Export NMF results")
        filemenu.AppendSeparator()
        menuExit=filemenu.Append(wx.ID_EXIT, "&Exit"," Close the program")
        
        #Setup data process menu.
        datamenu=wx.Menu()
        
        menuCompress=datamenu.Append(wx.ID_ANY, "&Compression", " Reduce spectra resolution and reduce memory usage")#compress
        
        menuUSM=datamenu.Append(wx.ID_ANY, "&Unsharp Mask", "Increase local contrast of spectra")#compress
        
        
        #subMenu = wx.Menu()
        #menuCompress = subMenu.Append(wx.ID_ANY, "Compression ratio"," Reduce spectra resolution and reduce memory usage")
        
        
        
        
        
        
        
        
        
        
        
        #menuUSM=compressmenu.Append(wx.ID_ANY, "&Reduce Size"," Reduce spectra resolution and reduce memory usage")
        #menuSelect=compressmenu.Append(wx.ID_ANY, "&Reduce Size"," Reduce spectra resolution and reduce memory usage")
        
        
        
        
        
        
        #Setup nimfa menu.
        nmfmenu=wx.Menu()
        menuNimfa=nmfmenu.Append(wx.ID_ANY, "&Nimfa","nimfa")
        menuScikit=nmfmenu.Append(wx.ID_ANY, "&Scikit","scikit")
        
        #Setup scikit menu.
        toolmenu=wx.Menu()
        menuSelect=toolmenu.Append(wx.ID_ANY, "&Select"," Select a region")
        
        #colormap
        colormenu=wx.Menu()
        
        cpname=colormap.name
        for i in range(cpname.shape[0]):
            number=cpname[i][0]
            label=cpname[i][1]
            menuItem=colormenu.Append(wx.ID_ANY,str(number+'   '+label),'',wx.ITEM_RADIO)
            self.Bind(wx.EVT_MENU, self.OnColor, menuItem)
            
            
            '''
        def CreatMenuItem(self,menu,label,kind=wx.ITEM_RADIO):
            menuItem=menu.Append(-1,label)
            self.bind(wx.EVT_MENU, self.OnColor, menuItem)
        '''
        
        #Setup About menu.
        aboutmenu=wx.Menu()
        menuAbout=aboutmenu.Append(wx.ID_ABOUT,"&About","")
       
        
        
        
        
        #creat the manubar
        menuBar=wx.MenuBar()
        
        menuBar.Append(filemenu,"&File")
        menuBar.Append(datamenu,"&Process") 
        menuBar.Append(nmfmenu,"&NMF") 
        menuBar.Append(toolmenu,"&Tool")
        menuBar.Append(colormenu,"&Color")
        menuBar.Append(aboutmenu,"&About") 
        
        self.SetMenuBar(menuBar)  # Adding the MenuBar to the Frame content.
        
        
        
        # Events.
        self.Bind(wx.EVT_MENU, self.OnOpen, menuOpen)
        self.Bind(wx.EVT_MENU, self.OnSave, menuSave)
        self.Bind(wx.EVT_MENU, self.OnExit, menuExit)
        self.Bind(wx.EVT_MENU, self.OnCompress, menuCompress)
        self.Bind(wx.EVT_MENU, self.OnUSM, menuUSM)
        self.Bind(wx.EVT_MENU, self.OnAbout, menuAbout)
        
        self.Bind(wx.EVT_MENU, self.OnNimfa, menuNimfa)
        self.Bind(wx.EVT_MENU, self.OnScikit, menuScikit)
        
        self.Bind(wx.EVT_MENU, self.OnSelect, menuSelect)
        '''
        self.sizer2 = wx.BoxSizer(wx.HORIZONTAL)
        self.buttons = []
        for i in range(0, 6):
            self.buttons.append(wx.Button(self, -1, "Button &"+str(i)))
            self.sizer2.Add(self.buttons[i], 1, wx.EXPAND)
        '''
        '''
        # Use some sizers to see layout options
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.control, 1, wx.EXPAND)
        self.sizer.Add(self.sizer2, 0, wx.EXPAND)
   
        #Layout sizers
        self.SetSizer(self.sizer)
        self.SetAutoLayout(1)
        self.sizer.Fit(self)
        self.Show()
        '''
        
        
    #define button 
    #menu
    def OnOpen(self,event):
        dlg = wx.FileDialog(self, "Load fit/npy file", "", "", wildcard="ARPES file (.fits;.npy)|*.fits;*.npy", style = wx.FD_OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            
            self.filename = dlg.GetFilename()
            self.dirname = dlg.GetDirectory()
            dataurl = join(self.dirname, self.filename)
            self.read(dataurl)
        
        dlg.Destroy()

    #menu save    
    def OnSave(self,event):
        
        
        dlg = wx.FileDialog(self, "Load fit/npy file", "", "", wildcard="NPY file (.npy)|*.npy" ,style = wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if dlg.ShowModal() == wx.ID_OK:
            
            self.filename = dlg.GetFilename()
            self.dirname = dlg.GetDirectory()
            saveurl = join(self.dirname, self.filename)
            self.save(saveurl)
            
        dlg.Destroy()
        
    #menu about    
    def OnAbout(self,event):
        dlg=wx.MessageDialog(self, "An NMF package for ARPES data analysis","About this software", wx.OK)
        dlg.ShowModal()
        dlg.Destroy()

    #menu exit
    def OnExit(self,event):
        self.Close(True)
    
    #menu compress
    def OnCompress(self,event):
        dlg = wx.NumberEntryDialog(self,'Reduce spectra resolution by',"compression ratio:","Compression Ratio",
                                    value=4, min=1, max=99)
        if dlg.ShowModal()== wx.ID_OK:
            cnum = dlg.GetValue()
            self.dataNew=self.compress(self.dataNew,cnum)
            MainWindow.dataNew=self.dataNew
            
        dlg.Destroy()
        
    #menu unsharp mask
    def OnUSM(self,event):
        dlg = USMSetup(parent=None)
        if dlg.ShowModal() == wx.ID_OK:
            try:
                self.dataNew=usm(self.dataNew,dlg.USM_radius,dlg.USM_precent,dlg.USM_threhold)
                MainWindow.dataNew=self.dataNew
                
            except AttributeError:
                print("error")
                dlg=wx.MessageDialog(self, "No data loaded.", caption='Error!',
                style=wx.ICON_ERROR)
                dlg.ShowModal()
                dlg.Destroy()
            
        
    def OnNimfa(self,event):
        #MainWindow.Nimfa=NMFAuiWindow.nimfa(self)
        #MainWindow.Nimfa=NMFAuiWindow(parent=self,name='nimfa',size=(800,600))
        try:
            nmfpanes = NMFAuiWindow._manager.GetAllPanes()
            for pane in nmfpanes:
                NMFAuiWindow._manager.ClosePane(pane)
            
        except:
            pass
        NMFAuiWindow.nimfa(self)


    def OnScikit(self,event):
        try:
            nmfpanes = NMFAuiWindow._manager.GetAllPanes()
            for pane in nmfpanes:
                NMFAuiWindow._manager.ClosePane(pane)
            
        except:
            pass
        NMFAuiWindow.scikit(self)       


    
    
    #menu color    
    def OnColor(self,event):
    
        dlg = wx.NumberEntryDialog(self,'Reduce spectra resolution by',"compression ratio:","Compression Ratio",
                                    value=4, min=1, max=99)
        if dlg.ShowModal()== wx.ID_OK:
            cnum = dlg.GetValue()
            self.dataNew=self.compress(self.dataNew,cnum)
            MainWindow.dataNew=self.dataNew
            
        dlg.Destroy()
    
    def OnSelect(self,event):
        MainWindow.Select=True
    

    #Function region
    
    #load file    
    def read(self,dataurl):
        if dataurl.endswith('.fits'):
            hdulist = fits.open(dataurl,ignore_missing_end=True)
            
            hdu = hdulist[1]
            
            MainWindow.mapx=hdulist[0].header['N_0_0']
            MainWindow.mapy=hdulist[0].header['N_0_1']
            
            array = hdu.data
            dataset = np.empty((array.shape[0],array[0][-1].shape[0],array[0][-1].shape[1]))
            for i in range(array.shape[0]):
                dataset[i] = np.array(array[i][-1])
            self.dataNew=dataset
            MainWindow.dataNew=self.dataNew            
            
            
        if dataurl.endswith('.npy'):
            dataset = np.load(dataurl, allow_pickle=True)   
            self.dataNew = dataset[0]
            MainWindow.mapx = dataset[1]
            MainWindow.mapy = dataset[2]
            MainWindow.dataNew=self.dataNew
            
            
   #save data file
    def save(self, saveurl):
        savedata=np.array([MainWindow.dataNew,MainWindow.mapx,MainWindow.mapy])
        np.save(saveurl, savedata)



    #compress data
    def compress(self,data,snum):
    
        try:
            dataset = np.empty((data.shape[0],int(data.shape[1]/snum),int(data.shape[2]/snum)))
            for i in range(data.shape[0]):
                img = Image.fromarray(data[i])
                img = img.resize((int(data.shape[2]/snum),int(data.shape[1]/snum)))
                array_image = np.asarray(img)
                dataset[i] = array_image
                
            return dataset
        except:
            print(sys.exc_info()[0])
        
        
        
        
    #unsharp masking    

        
          
class nimfaRun():
    def __init__(self,data):
    #nimfa run
    # read ARPES fit  data
    # preprocess ARPES data matrix
    #V, shrink_size = preprocess(data)
    # run factorization from nimfa package
        nimfaRun.W, nimfaRun.H= self.factorize_nimfa(data)
        

    def factorize_nimfa(self,V):
        """
        Perform standard NMF factorization from nimfa package. 
    
        Return basis and mixture matrices of the fitted factorization model. 

        """

        V = V.reshape(V.shape[0],(V.shape[1]*V.shape[2]))
        
        
        if np.any(V<0):
            V[V<0]=0
        nmf = nimfa.Nmf(V, seed="random_c", rank=nimfapanel.rank, n_run=nimfapanel.nrun, update=nimfapanel.update, objective=nimfapanel.objective, max_iter=nimfapanel.maxiter)
        fit = nmf()

        W = fit.basis()
        H = fit.coef()
        return W, H

    
    

class scikitRun():
    def __init__(self,data):
    #nimfa run
    # read ARPES fit  data
    # preprocess ARPES data matrix
    #V, shrink_size = preprocess(data)
    # run factorization from nimfa package
        scikitRun.W, scikitRun.H= self.factorize_scikit(data)
        
    
    def factorize_scikit(self,V):
        """
        Perform NMF factorization on the ORL faces data matrix. 
    
        Return basis and mixture matrices of the fitted factorization model. 
    
        :param V: The ORL faces data matrix. 
        :type V: `numpy.matrix`
        """    
        V = V.reshape(V.shape[0],(V.shape[1]*V.shape[2]))
    
        if np.any(V<0):
            V[V<0]=0
        nmf = NMF(n_components=scikitpanel.component,init=scikitpanel.initialize,beta_loss=scikitpanel.betaloss,tol=scikitpanel.tolerance,max_iter=scikitpanel.iteration)
        W = nmf.fit_transform(V)
        H = nmf.components_  
        return W, H



#Unsharp mask dialog
class USMSetup(wx.Dialog):
    def __init__(self, parent):
        wx.Dialog.__init__(self, parent, -1, "Unshark Mask", size= (160,220))
        self.panel = wx.Panel(self,wx.ID_ANY)
        
        self.radiustitle = wx.StaticText(self.panel, label="Radius: ", pos=(20,20))
        self.radius = wx.lib.intctrl.IntCtrl(self.panel, value=3, pos=(80,16), size=(40,-1))
        self.percenttitle = wx.StaticText(self.panel, label="Percent: ", pos=(20,60))
        self.percent = wx.lib.intctrl.IntCtrl(self.panel, value=150, pos=(80,56), size=(40,-1))
        self.threholdtitle = wx.StaticText(self.panel, label="Title: ", pos=(20,100))
        self.threhold = wx.lib.intctrl.IntCtrl(self.panel, value=5, pos=(80,96), size=(40,-1))
        
        self.OKButton =wx.Button(self.panel,wx.ID_OK, label="OK", pos=(20,140),size=(50,-1))
        self.closeButton =wx.Button(self.panel,wx.ID_EXIT, label="Cancel", pos=(80,140),size=(50,-1))
        self.Bind(wx.EVT_BUTTON, self.EnterValue, self.OKButton)
        self.Bind(wx.EVT_BUTTON, self.OnQuit, self.closeButton)
        self.Bind(wx.EVT_CLOSE, self.OnQuit)
        
        #self.Show()    
        
    def OnQuit(self, event):
        self.SetReturnCode(wx.ID_EXIT)
        self.Destroy()
   
    def EnterValue(self,event):
        self.USM_radius = self.radius.GetValue()
        self.USM_precent = self.percent.GetValue()
        self.USM_threhold = self.threhold.GetValue()
        self.SetReturnCode(wx.ID_OK)
        self.Destroy()
        #try:
        
#Unsharp mask        
class usm():
    def __init__(self,data,radius,percent,threshold):
        
        try:
            
            for i in range(data.shape[0]):
                img = Image.fromarray(data[i])
                img = img.convert('RGB')
                img_f = img.filter(ImageFilter.UnsharpMask(radius, percent, threshold))
                img_f = img_f.convert('P')
                array_image = np.asarray(img_f)
                data[i]=array_image
        except:
            print("error")


class colormap():
    name=np.loadtxt(join(dirname(__file__),'colorname.csv'),delimiter=',',dtype=str)


class NMFAuiWindow(wx.Window):
    def __init__(self, parent, name, *args, **kwargs):
        wx.Window.__init__(self, parent, -1, style=wx.CLIP_CHILDREN|FULL_REPAINT_ON_RESIZE, **kwargs)
        self.SetMinSize((600, 200))
        
        # Create an AUI Manager and tell it to manage this Frame
        #NMFAuiWindow._manager = aui.AuiManager()
        #NMFAuiWindow._manager.SetManagedWindow(self)
        #NMFAuiWindow._manager.Update()
        
    def nimfa(self):
        # Create an AUI Manager and tell it to manage this Frame    
        NMFAuiWindow._manager = aui.AuiManager()
        NMFAuiWindow._manager.SetManagedWindow(self)


        nmf=nimfapanel(parent=self)

        control_info = aui.AuiPaneInfo().Top().Name('Nimfa Parameters').Caption('Nimfa Parameters').Resizable(True).\
            CloseButton(True).MaximizeButton(False).MinimizeButton(False).Show().Floatable(False).DestroyOnClose(b=True).MinSize((600,42))


        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(nmf, 1, wx.EXPAND)
        self.SetSizer(sizer,deleteOld=True)

        NMFAuiWindow._manager.AddPane(nmf, control_info)
        #self._manager.AddPane(plotter, plot_info)
        NMFAuiWindow._manager.Update()
        
        
    def scikit(self):

        # Create an AUI Manager and tell it to manage this Frame
        NMFAuiWindow._manager = aui.AuiManager()
        NMFAuiWindow._manager.SetManagedWindow(self)
        #sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        nmf=scikitpanel(parent=self)
        
        #sizer.Add(nmf, 1, wx.EXPAND)
        #self.SetSizer(sizer,deleteOld=True)

        control_info = aui.AuiPaneInfo().Top().Name('Scikit Parameters').Caption('Scikit Parameters').\
            CloseButton(True).MaximizeButton(False).MinimizeButton(False).Show().Floatable(False).DestroyOnClose(b=True).MinSize((600,42))
        
        NMFAuiWindow._manager.AddPane(nmf, control_info)
        NMFAuiWindow._manager.Update()    
    

    def plot(self,W,H):
        

        plotter1 = PlotNotebook(parent=self, size=(1000,600))
        plotter2 = PlotNotebook(parent=self, size=(1000,600))
        #toolbar=aui.AuiToolBar(self, -1, wx.DefaultPosition, wx.DefaultSize, agwStyle=aui.AUI_TB_DEFAULT_STYLE | aui.AUI_TB_OVERFLOW | aui.AUI_TB_TEXT | aui.AUI_TB_HORZ_TEXT)
        
        sizer1 = wx.BoxSizer(wx.HORIZONTAL)
        sizer2 = wx.BoxSizer(wx.HORIZONTAL)
        #'''
        def fig(W,H):
            s = np.zeros((W.shape[1],W.shape[1]))
            W_norm = np.zeros((W.shape))
            H_norm = np.zeros((H.shape))
            for i in range(s.shape[1]):
                s[i,i] = np.max(W[:,i])
            s_inv = np.linalg.inv(s)
            W_norm = W.dot(s_inv)
            H_norm = s.dot(H)
                
            #plt.figure(figsize=(12,5*H.shape[0]))
            for i in range(H.shape[0]):
                axes1 = plotter1.add(parent=self,name='Spectrum'+str(i+1)).gca()
                img1=axes1.imshow(H_norm[i].reshape(MainWindow.dataNew.shape[1], MainWindow.dataNew.shape[2]), origin='lower', cmap='plasma')
                Plot.figure.colorbar(img1, ax=axes1)
                plt.savefig(join(dirname(__file__),"temp","spectra","spectrum"+str(i+1)+".png"))    
            
            for i in range(H.shape[0]):
                #plt.subplot(1,2,2)
                axes2 = plotter2.add(parent=self,name='Map'+str(i+1)).gca()
                img2=axes2.imshow(W_norm[:,i].reshape(MainWindow.mapy,MainWindow.mapx), origin='lower', cmap='plasma')
                Plot.figure.colorbar(img2, ax=axes2)
                plt.savefig(join(dirname(__file__),"temp","mapping","map"+str(i+1)+".png"))        


        '''
        axes1 = plotter.add(parent=self,name='figure 1').gca()
        axes1.plot([1, 2, 3], [2, 1, 4])
        axes2 = plotter.add(parent=self,name='figure 2').gca()
        axes2.plot([1, 2, 3, 4, 5], [2, 1, 4, 2, 3])
        '''       
        
        fig(W,H)
        sizer1.Add(plotter1, 1, wx.EXPAND)
        sizer2.Add(plotter2, 1, wx.EXPAND)
        self.SetSizer(sizer1)
        self.SetSizer(sizer2)
        
        plot1_info = aui.AuiPaneInfo().Name('NMF spectrum').Caption('NMF spectrum').Top().Row(1).BestSize(600,600).\
        CloseButton(True).MaximizeButton(True).MinimizeButton(True).Show().Floatable(True).DestroyOnClose(b=True).MinSize((600,600)).Float()
        
        plot2_info = aui.AuiPaneInfo().Name('NMF mapping').Caption('NMF mapping').Top().Row(2).BestSize(600,600).\
        CloseButton(True).MaximizeButton(True).MinimizeButton(True).Show().Floatable(True).DestroyOnClose(b=True).MinSize((600,600)).Float()
        
        NMFAuiWindow._manager.AddPane(plotter1, plot1_info)
        NMFAuiWindow._manager.AddPane(plotter2, plot2_info)
        #NMFAuiWindow._manager.AddPane(toolbar, aui.AuiPaneInfo().ToolbarPane().Bottom().Name('toolbar').Caption("Toolbar"))
        NMFAuiWindow._manager.Update()           
    
        
    def __OnQuit(self, event):
        NMFAuiWindow._manager.UnInit()
        del NMFAuiWindow._manager
        NMFAuiWindow.Destroy()


            
class nimfapanel(wx.Panel):
    def __init__(self,parent, id=-1): 
        wx.Panel.__init__(self, parent,id=wx.ID_ANY)
            
        
            
        fgs=wx.GridBagSizer(2,8)
        
        self.ranktxt = wx.StaticText(self, label="Rank",style=wx.EXPAND|wx.ALIGN_CENTER)
        self.nruntxt = wx.StaticText(self, label="# of Run",style=wx.EXPAND|wx.ALIGN_CENTER)
        self.updatetxt = wx.StaticText(self, label="Update",style=wx.EXPAND|wx.ALIGN_CENTER)
        self.objectivetxt = wx.StaticText(self, label="Objective",style=wx.EXPAND|wx.ALIGN_CENTER)
        self.maxitertxt = wx.StaticText(self, label="Max Iteration",style=wx.EXPAND|wx.ALIGN_CENTER)
        self.runButton =wx.Button(self,wx.ID_OK, label="RUN",style=wx.EXPAND|wx.CENTER,size=(80,-1))
        self.resetButton =wx.Button(self,wx.ID_ANY, label="RESET",style=wx.EXPAND|wx.CENTER,size=(80,-1))
        self.plotButton =wx.Button(self,wx.ID_ANY, label="PLOT",style=wx.EXPAND|wx.CENTER,size=(80,-1))
        
        self.rank = wx.lib.intctrl.IntCtrl(self, value=3,style= wx.ALIGN_RIGHT,size=(40,-1),min=1)
        self.nrun = wx.lib.intctrl.IntCtrl(self, value=150,style= wx.ALIGN_RIGHT,size=(40,-1),min=1)
        self.update=wx.Choice(self, -1,choices=['euclidean','divergence'])
        self.update.SetSelection(0)
        self.objective=wx.Choice(self, -1,choices=['fro','div','conn'])
        self.objective.SetSelection(0)
        self.maxiter = wx.lib.intctrl.IntCtrl(self, value=5,style= wx.ALIGN_RIGHT,size=(60,-1),min=0)

        fgs.Add(self.ranktxt,pos=(0,0),flag=wx.ALL|wx.EXPAND)
        fgs.Add(self.nruntxt,pos=(0,1),flag=wx.ALL|wx.EXPAND)
        fgs.Add(self.updatetxt,pos=(0,2),flag=wx.EXPAND, border=15)
        fgs.Add(self.objectivetxt,pos=(0,3),flag=wx.EXPAND, border=5)
        fgs.Add(self.maxitertxt,pos=(0,4),flag=wx.EXPAND)
        fgs.Add(self.runButton,span=(2,1),pos=(0,5),flag=wx.EXPAND)
        fgs.Add(self.resetButton,span=(2,1),pos=(0,6),flag=wx.EXPAND)
        fgs.Add(self.plotButton,span=(2,1),pos=(0,7),flag=wx.EXPAND)
        fgs.Add(self.rank,pos=(1,0),flag=wx.EXPAND)
        fgs.Add(self.nrun,pos=(1,1),flag=wx.EXPAND)
        fgs.Add(self.update,pos=(1,2),flag=wx.EXPAND)
        fgs.Add(self.objective,pos=(1,3),flag=wx.EXPAND)
        fgs.Add(self.maxiter,pos=(1,4),flag=wx.EXPAND)
        
        for i in range(8):
            fgs.AddGrowableCol(i)
        #fgs.AddGrowableRow(1)
        
        fgs.SetSizeHints(self)
        self.SetSizer(fgs)
        
        self.Bind(wx.EVT_BUTTON, self.PlotResult, self.plotButton)
        self.Bind(wx.EVT_BUTTON, self.OnRun, self.runButton)
        self.Bind(wx.EVT_BUTTON, self.OnReset, self.resetButton)

    def OnRun(self, event):
        nimfapanel.rank=self.rank.GetValue()
        nimfapanel.nrun=self.nrun.GetValue()
        nimfapanel.update=self.update.GetString(self.update.GetCurrentSelection())
        nimfapanel.objective=self.objective.GetString(self.objective.GetCurrentSelection())
        nimfapanel.maxiter=self.maxiter.GetValue()
        try:
            nimfaRun(MainWindow.dataNew)
            
        except AttributeError:
            dlg=wx.MessageDialog(self, "No data loaded.", caption='Error!',
                style=wx.ICON_ERROR)
            dlg.ShowModal()
            dlg.Destroy()
            
    def OnReset(self,event):
        self.rank.SetValue(3)
        self.nrun.SetValue(150)
        self.update.SetSelection(0)
        self.objective.SetSelection(0)
        self.maxiter.SetValue(5)
   
    def PlotResult(self,event):
        try:
            NMFAuiWindow.plot(self, nimfaRun.W, nimfaRun.H)
        except AttributeError as error:
            print(Error)
            dlg=wx.MessageDialog(self, "No data loaded.", caption='Error!',
                style=wx.ICON_ERROR)
            dlg.ShowModal()
            dlg.Destroy()


class scikitpanel(wx.Panel):
    def __init__(self,parent, id=-1): 
        wx.Panel.__init__(self, parent,id=wx.ID_ANY)
            
        
            
        fgs=wx.GridBagSizer(2,8)
        
        self.componenttxt = wx.StaticText(self, label="Component",style=wx.EXPAND|wx.ALIGN_CENTER)
        self.tolerancetxt = wx.StaticText(self, label="Tolerance",style=wx.EXPAND|wx.ALIGN_CENTER)
        self.betalosstxt = wx.StaticText(self, label="Beta loss",style=wx.EXPAND|wx.ALIGN_CENTER)
        self.initializetxt = wx.StaticText(self, label="Initialize",style=wx.EXPAND|wx.ALIGN_CENTER)
        self.itertxt = wx.StaticText(self, label="Iteration",style=wx.EXPAND|wx.ALIGN_CENTER)
        self.runButton =wx.Button(self,wx.ID_OK, label="RUN",style=wx.EXPAND|wx.CENTER,size=(80,-1))
        self.resetButton =wx.Button(self,wx.ID_ANY, label="RESET",style=wx.EXPAND|wx.CENTER,size=(80,-1))
        self.plotButton =wx.Button(self,wx.ID_ANY, label="PLOT",style=wx.EXPAND|wx.CENTER,size=(80,-1))
        
        self.component = wx.lib.intctrl.IntCtrl(self, value=5,style= wx.ALIGN_RIGHT,size=(40,-1),min=0)
        self.tolerance = wx.TextCtrl(self, value='1E-4',validator=MyNumberValidator(),style= wx.ALIGN_RIGHT,size=(40,-1))
        self.initialize=wx.Choice(self, -1,choices=['None','random','nndsvd','nndsvda','nndsvdar'])
        self.initialize.SetSelection(0)
        self.betaloss=wx.Choice(self, -1,choices=['frobenius','kullback-leibler','itakura-saito'])
        self.betaloss.SetSelection(0)
        self.iteration = wx.lib.intctrl.IntCtrl(self, value=200,style= wx.ALIGN_RIGHT,size=(60,-1),min=0)

        fgs.Add(self.componenttxt,pos=(0,0),flag=wx.ALL|wx.EXPAND)
        fgs.Add(self.tolerancetxt,pos=(0,1),flag=wx.ALL|wx.EXPAND)
        fgs.Add(self.betalosstxt,pos=(0,2),flag=wx.EXPAND, border=15)
        fgs.Add(self.initializetxt,pos=(0,3),flag=wx.EXPAND, border=5)
        fgs.Add(self.itertxt,pos=(0,4),flag=wx.EXPAND)
        fgs.Add(self.runButton,span=(2,1),pos=(0,5),flag=wx.EXPAND)
        fgs.Add(self.resetButton,span=(2,1),pos=(0,6),flag=wx.EXPAND)
        fgs.Add(self.plotButton,span=(2,1),pos=(0,7),flag=wx.EXPAND)
        fgs.Add(self.component,pos=(1,0),flag=wx.EXPAND)
        fgs.Add(self.tolerance,pos=(1,1),flag=wx.EXPAND)
        fgs.Add(self.initialize,pos=(1,2),flag=wx.EXPAND)
        fgs.Add(self.betaloss,pos=(1,3),flag=wx.EXPAND)
        fgs.Add(self.iteration,pos=(1,4),flag=wx.EXPAND)
        
        for i in range(8):
            fgs.AddGrowableCol(i)
        #fgs.AddGrowableRow(1)
        
        fgs.SetSizeHints(self)
        self.SetSizer(fgs)
        
        self.Bind(wx.EVT_BUTTON, self.PlotResult, self.plotButton)
        self.Bind(wx.EVT_BUTTON, self.OnRun, self.runButton)
        self.Bind(wx.EVT_BUTTON, self.OnReset, self.resetButton)

    def OnRun(self, event):
        scikitpanel.component=self.component.GetValue()
        scikitpanel.tolerance=float(self.tolerance.GetLineText(0))
        scikitpanel.initialize=self.initialize.GetString(self.initialize.GetCurrentSelection())
        scikitpanel.initialize=None if scikitpanel.initialize=='None' else scikitpanel.initialize  #Initialise default input is Empty-None
        scikitpanel.betaloss=self.betaloss.GetString(self.betaloss.GetCurrentSelection())
        scikitpanel.iteration=self.iteration.GetValue()
        try:
            scikitRun(MainWindow.dataNew)
            
        except AttributeError as error:
            print(error)
            dlg=wx.MessageDialog(self, "No data loaded.", caption='Error!',
                style=wx.ICON_ERROR)
            dlg.ShowModal()
            dlg.Destroy()
            
    def OnReset(self,event):
        self.component.SetValue(5)
        self.tolerance.SetValue('1E-4')
        self.initialize.SetSelection(0)
        self.betaloss.SetSelection(0)
        self.iteration.SetValue(200)            
   
    def PlotResult(self,event):
        try:
            NMFAuiWindow.plot(self,scikitRun.W, scikitRun.H)
        except AttributeError as error:
            print(error)
            dlg=wx.MessageDialog(self, "No data loaded.", caption='Error!',
                style=wx.ICON_ERROR)
            dlg.ShowModal()
            dlg.Destroy()        
        
        
        
        
class Plot(wx.Panel):
    def __init__(self, parent, id=-1, dpi=None, **kwargs):
        wx.Panel.__init__(self, parent, id, **kwargs)
        Plot.figure,Plot.ax= plt.subplots(dpi=dpi, figsize=(6, 5))
        #plt.figure(dpi=dpi, figsize=(2, 2))
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.toolbar = NavigationToolbar(self.canvas)
        self.toolbar.Realize()
        
        self.toggle_selector_RS = RectangleSelector(Plot.ax, self.onselect, drawtype='box', button=1,
                                        #minspanx=5, minspany=5, 
                                        spancoords='data', interactive=True 
                                        )
        Plot.figure.canvas.mpl_connect('key_press_event', self.toggle_selector)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas, 1, wx.EXPAND)
        sizer.Add(self.toolbar, 0, wx.LEFT | wx.EXPAND)
        self.SetSizer(sizer)
    
    def onselect(self,eclick, erelease):
        #"eclick and erelease are matplotlib events at press and release."
        print('startposition: (%f, %f)' % (eclick.xdata, eclick.ydata))
        print('endposition  : (%f, %f)' % (erelease.xdata, erelease.ydata))
        print('used button  : ', eclick.button)
        self.coordinate=self.toggle_selector_RS.geometry
        self.coordinate=self.coordinate.astype(int)
        print(self.coordinate)

    def toggle_selector(self,event):
        print('Key pressed.')
        if event.key in ['Q', 'q'] and self.toggle_selector_RS.active:
            print('RectangleSelector deactivated.')
            self.toggle_selector_RS.set_active(False)
            MainWindow.Select=False
            
        if MainWindow.Select and not self.toggle_selector_RS.active:
            print('RectangleSelector activated.')
            self.toggle_selector_RS.set_active(True)
        if event.key == "ctrl+alt+a" and not self.toggle_selector_RS.active:
            print('RectangleSelector activated.')
            self.toggle_selector_RS.set_active(True)
            
        if event.key in ['s','S'] or event.key=='enter' and self.toggle_selector_RS.active:
            dlg = wx.MessageDialog(None, "Are you sure for the selection? \n(This will overwrite the previous result.)", "Select Region", wx.YES_NO | wx.ICON_QUESTION)
            if dlg.ShowModal() == wx.ID_YES:
                print("crop image")
                print(MainWindow.dataNew.shape)
                data=MainWindow.dataNew.reshape((MainWindow.mapy,MainWindow.mapx,MainWindow.dataNew.shape[1],MainWindow.dataNew.shape[2]))
                data=data[min(self.coordinate[1,:]):max(self.coordinate[1,:]),min(self.coordinate[:,1]):max(self.coordinate[:,1]),:,:]
                MainWindow.mapy=max(self.coordinate[1,:])-min(self.coordinate[1,:])
                MainWindow.mapx=max(self.coordinate[:,1])-min(self.coordinate[:,1])
                data=data.reshape(MainWindow.mapy*MainWindow.mapx,data.shape[-2],data.shape[-1])
                MainWindow.dataNew=data
                
                self.Close(True)
                print(MainWindow.dataNew.shape)
            dlg.Destroy()

            
            
        rectprops = dict(facecolor='red', edgecolor = 'black',
                     alpha=0.5, fill=True)
                     
                     
                     

class PlotNotebook(wx.Panel):
    def __init__(self, parent, id=-1,**kwargs):
        wx.Panel.__init__(self, parent, id=id,**kwargs)
        self.nb = aui.AuiNotebook(self)

        
        sizer = wx.BoxSizer()
        sizer.Add(self.nb, 1, wx.EXPAND)
        self.SetSizer(sizer)
        sizer.SetSizeHints(self)
        

    def add(self,parent, name="plot"):
        page = Plot(self.nb)
        self.nb.AddPage(page, name)
        return page.figure



app=wx.App()


main = MainWindow(None,"NMF")


main.Show(True)






    

app.MainLoop()