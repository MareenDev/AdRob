from classes.framework import Interpreter, JSONHandler, PathHandler
from torch import Tensor
import numpy as np
from torchvision import utils
from os import path
import matplotlib.pyplot as plt

class AdvExampleGrids(Interpreter):    
    def __init__(self, refname, **kwargs):
        super().__init__(refname, **kwargs)
        self.name = "Grids" 
        self.description = "List of Grids of adversarial examples" 
        self.inputs= kwargs["data"]
        
        self.data = dict()
        self._calculate()

    def get(self):
        return self.data
    
    def setData(self, data):
        self.data = data

    def _calculate(self):
        for  attackref, value in self.inputs.items():
            tl = [] 
            for l in value:
                for arr in l:
                    tl.append(Tensor(np.transpose(arr, (2, 0, 1)).astype(np.float32)))
            self.data[attackref] = utils.make_grid(tensor=tl,nrow=len(tl)//len(value))

        """paths = AttackProtocol(self.refname).getAttackPaths()

        for p in paths:
            ph = PathHandler(p)
            filenames = ph.get_filenames("npy")
            loadedData=[np.load(path.join(p,filename)) for p,filenameList in filenames.items() for filename in filenameList ]
            tensorlist = [Tensor(np.transpose(array, (2, 0, 1)).astype(np.float32)) for it in loadedData for array in it]
            self.data[path.basename(p)]=utils.make_grid(tensor=tensorlist,nrow=len(tensorlist)//len(loadedData))
        """

    def save(self, folder="./data/interpretation"):
    
        for key,value in self.data.items():
            ph = PathHandler(folder)
            subdir = ph.create_subdirectory(self.refname)
            filename = path.join(subdir,self.name+"_"+path.basename(key)+".png")
            utils.save_image(tensor=value,fp=path.join(filename))


class BarPlot(Interpreter):
    def __init__(self, refname, **kwargs):
        super().__init__(refname, **kwargs)
        plt.ioff()
        try:
            self.categories = kwargs["categories"]
        except:
            raise ValueError("Parameter 'categories' is missing.")
        
        try:
            self.counts =  kwargs["counts"]
        except:
            raise ValueError("Parameter 'counts' is missing.")
            
        self.colors =  ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf','#bccd22', '#17ff1f','#baca21', '#18cc1f','#8d564c', '#7e37c5', '#11f34e', '#33cab3','#1c22f5', '#f12ffe']
        
        colors = []
        for i,_ in enumerate(self.categories):

            colors.append(self.colors[i])
           # labels.append("tab:"+str(self.colors[i]))
        

        self.fig, self.ax = plt.subplots()
        p = self.ax.bar(self.categories, self.counts, color=colors)
        self.ax.set_ylabel('Clean Accuracy [%]')
        self.ax.set_xlabel('Modelle')
        self.ax.set_title('Compare Clean Accuracy')
        self.name = refname
        #self.ax.legend(title='')
        self.ax.bar_label(p)

    def save(self, folder):
        pH = PathHandler(folder)
        #subdir = pH.create_subdirectory(self.refname)
        filename_figure = path.join(folder,self.name+".jpg")

        self.fig.savefig(filename_figure) 

    def show(self):
        self.fig.show()