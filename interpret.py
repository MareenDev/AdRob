from classes.framework import Interpreter, JSONHandler, AttackProtocol, PathHandler, MultiPlot, AttackDataLoader
from os import path
import numpy as np
from torch import Tensor
from torchvision import utils

# Interpretations of attack with specific configuration
#TBD: Logik in ordner classes verschieben
class MeandAdvQueryCounts(Interpreter):
    def __init__(self, refname, **kwargs):
        super().__init__(refname, **kwargs)
        self.name = "Calls" 
        self.description = "Mittlere Anzahl der Schnittstellenaufrufe adversarialer Beispiele"         
        self.refname = refname
        try:
            self.queryCounts = kwargs["queryCounts"] 
        except :
            raise ValueError("Parameter 'queryCounts' is missing.")
        try:
            self.output = kwargs["output"] 
        except :
            raise ValueError("Parameter 'output' is missing.")
        
        self.data = self._calculate()

    def _calculate(self):
        counts=0    
        for key in self.queryCounts:
            for i in range(1,len(self.output[key])): 
                if self.output[key][i] != self.output[key][0]: #loadedC[key][0] contains the class of the original image
                    counts += int(self.queryCounts[key][i])#sum([int(y) for y in value[1:]])
        return np.mean(counts)

    def get(self):
        return self.data

    def save(self, folder="./data/interpretation"):
        jh = JSONHandler()
        filenameabs = path.join(folder,self.refname,self.name+".json")
        j = dict()
        j[self.refname]=self.data

        jh.setData(j)
        jh.save(filename=filenameabs)


class Accuracy(Interpreter):

    def __init__(self, refname, **kwargs):
        super().__init__(refname, **kwargs)
        self.name = "Akkuranz" 
        self.description = "Akkuranz" 
        try:
            self.labels = kwargs["labels"] 
        except :
            raise ValueError("Parameter 'labels' is missing.")
        try:
            self.predictions = kwargs["predictions"] 
        except :
            raise ValueError("Parameter 'predictions' is missing.")

        self.data = self._calculate()

    def _calculate(self):
        total=0
        correct=0
        for i in range(len(self.predictions)):

            total += 1
            if self.predictions[i] == self.labels[i]:
                correct +=1
        return correct/total

    def get(self):
        return self.data

    def save(self, folder="./data/interpretation"):
        jh = JSONHandler()
        filenameabs = path.join(folder,self.refname,self.name+".json")
        j = dict()
        j[self.refname]=self.data

        jh.setData(j)
        jh.save(filename=filenameabs)

class MeanDistortionLp(Interpreter):

    def __init__(self, refname, **kwargs):
        super().__init__(refname, **kwargs)
        try:
            self.referencesData = kwargs["original"] 
        except :
            raise ValueError("Parameter 'original' is missing.")
        try:
            self.adversarialData = kwargs["adversarials"] 
        except :
            raise ValueError("Parameter 'adversarials' is missing.")
        try:
            self.ord = kwargs["ord"] 
        except :
            raise ValueError("Parameter 'ord' is missing.")
        self.name = "L"+str(self.ord)
        self.description = "L"+str(self.ord)+" Metrik"

        self._calculate()

    def get(self):
        return self.data

    def _calculate(self):
        dist = []
        for i in range(len(self.referencesData)):
            perturbation = np.reshape(self.adversarialData[i] - self.referencesData[i], [-1])
            dist.append(np.linalg.norm(x=perturbation,ord=self.ord))
        if len(dist) == 0:
            dist =[0]

        self.data = np.mean(dist)
 
    def save(self, folder="./data/interpretation"):
        jh = JSONHandler()
        filenameabs = path.join(folder,self.refname,self.name+".json")
        j = dict()
        j[self.refname]=self.data

        jh.setData(j)
        jh.save(filename=filenameabs)


class DeepFoolMeasure(MeanDistortionLp):
    def _calculate(self):
        dist = []
        for i in range(len(self.referencesData)):
            perturbation = np.reshape(np.subtract(self.adversarialData[i], self.referencesData[i]), [-1])
            dist.append(np.linalg.norm(x=perturbation,ord=np.inf)/np.linalg.norm(x=np.reshape(self.referencesData[i],[-1]),ord=self.ord))
        if len(dist) == 0:
            dist =[0]

        self.data = np.mean(dist)


class Plot(Interpreter):
    def __init__(self, refname, **kwargs):
        super().__init__(refname, **kwargs)
        self.plot = None
        self.xSuffix = "" 
        self.ySuffix = ""  

        self.data = dict()
        self.refPredictions =  dict()
        self.adversarialPredictions =  dict()
        self.adversarialInputs = dict()
        self.refInputs =dict()
        self.attackQueryCalls = dict()
        self.labels = dict()
        for key,attackData in kwargs["data"].items():#Loop über die verschiedenen Angriffsausführungen
            self.adversarialPredictions[key] =[]
            self.refPredictions[key] =[]
            self.refInputs[key] =[]
            self.adversarialInputs[key] =[]
            self.attackQueryCalls[key] = []
            self.labels[key] =[]

            for k,data in attackData.items(): #Loop über die Inputelement-Referenzen
                for i,advPred in enumerate(data["output"]["advList"]):#abhängigkeit zu x_ref fehlt
                    self.adversarialPredictions[key].append(advPred)    
                    self.refPredictions[key].append(data["output"]["ref"])
                    self.adversarialInputs[key].append(data["input"]["advList"][i])   
                    self.refInputs[key].append(data["input"]["ref"])
                    self.labels[key].append(data["labels"])   
                    self.attackQueryCalls[key].append(data["queryCalls"]["advList"][i])   
                    
    def get(self):
        return self.data
    
    def createPlot(self,size,x_step,y_step,x_max,y_max,useLowerBound=False,useConfigNames=False ):
        """self.plot = MultiPlot(x_label=self.xSuffix,y_label=self.ySuffix,
                              size=size,x_step=x_step,y_step=y_step,x_max=x_max,y_max=y_max,title=self.description)
        
        self.plot.addData(name=path.basename(self.refname) ,x=self.data["x"],y=self.data["y"],useLowerBound=useLowerBound,
                          useConfigNames=useConfigNames,confignames=self.data["names"])"""

    def save(self, folder ="./data/interpretation/"):
        pH = PathHandler(folder)
        subdir = pH.create_subdirectory(self.refname)
        filename_figure = path.join(subdir,self.name+".jpg")
        filename_data = path.join(subdir,self.name+".json")

        self.plot.save(filename_figure) 
        jH = JSONHandler()
        savingData = dict()
        for key,value in self.data.items():
            savingData[key]=(str(value))
        jH.setData(savingData)
        jH.save(filename_data)
    
    def show(self):
        self.plot.show()

class RobustnessCurve(Plot):
    """In Anlehnung a
 https://github.com/niklasrisse/adversarial-examples-and-where-to-find-them/blob/master/generate_robustness_curves.py"""
    
    def __init__(self, refname, **kwargs):
        super().__init__(refname, **kwargs)    
        self.name = "RobustnessCurve" 
        self.description = "Verteilung von Bildpunkten anhand der Distortion" 
        self.xSuffix = "perturbation size"
        self.ySuffix = "T"
        try:
            self.norm = kwargs['norm'] 
        except :
            raise ValueError("Parameter 'norm' is missing.")
        
        self._calculate()

    def _calculate(self):
        for key in self.adversarialInputs: 
            dists_r = []
            advInput = np.array(self.adversarialInputs[key])
            refInput = np.array(self.refInputs[key])
            labels = np.array(self.labels[key])
            refPredictions = np.array(self.refPredictions[key])
            dists_r = np.array([np.linalg.norm(x = vector, ord = self.norm) for 
                                vector in np.subtract(advInput.reshape(advInput.shape[0],-1 ), 
                                refInput.reshape(refInput.shape[0], -1))])
            
            dists_r[refPredictions != labels] = 0 #Werte bei denen die die Prediktionen, nicht dem Datensatz entsprechen werden nicht berücksichtigt 
            dists_r.sort(axis=0)
            probs = 1/float(len(self.refPredictions[key])) * np.arange(1, len(self.refPredictions[key])+1)

            probs[np.isnan(dists_r)] = 1.0
            dists_r = np.nan_to_num(dists_r, nan = np.nanmax(dists_r))

            self.data[key]={"x" : dists_r, "y": probs }

    def createPlot(self,size,x_step,y_step,x_max,y_max,useLowerBound=False,usePoinLabels=False ):
        self.plot = MultiPlot(x_label=self.xSuffix,y_label=self.ySuffix,
                              size=size,x_step=x_step,y_step=y_step,x_max=x_max,y_max=y_max,title=self.description)
        for key,data in self.data.items(): 
            self.plot.addData(name=path.basename(key) ,x=data["x"],y=data["y"],useLowerBound=True)

# Implementierung in Anlehnung an https://doi.org/10.1109/ICPR48806.2021.9413143
class AccuracyPerturbationCurve(Plot):
    """Berechnung von Accuracy/Distortion zu einer (Kombination von) Angriffsausführung(en)"""
    def __init__(self, refname, **kwargs):
        super().__init__(refname, **kwargs)
        try:
            self.norm = kwargs['norm'] 
        except :
            raise ValueError("Parameter 'norm' is missing.")
        
        self.name = "AccuracyPerturbationCurve_L"+str(self.norm)
        self.description = "Scatterplot zu Accuracy-Perturbation-Curve "  

        self.xSuffix = "L"+str(self.norm)+"-Metrik"
        self.ySuffix = "Robuste Akkuranz"
        self._calculate()

    def _calculate(self):
        #1. Kalkulieren der (relativen) robusten Akkuranz über den gesamten Datensatz - D_all 
        #2. Herausfiltern der Datenpunkte, die durch den Klassifikator falsch abgebildet werden - D_interesse
        #3. Kalkulieren der mittleren Distortion entsprechend der DeepFool-Logik auf D_interesse
        self.data["x"] = []
        self.data["y"] = []

        for attackConfig in self.refInputs:
            #Zu 1.Absolute, robuste Akkuranz
            self.data["names"]=path.basename(attackConfig)
            self.data["y"].append(Accuracy(refname=self.refname,labels= self.labels[attackConfig],
                                    predictions=self.adversarialPredictions[attackConfig]).get())

            #Zu 2.Filterung der falsch klassifizierten Daten durch den Klassifikator 
            indizeList = []
            refInputs = []
            advInput = []

            for i in range(len(self.refPredictions[attackConfig])):
                if self.refPredictions[attackConfig][i]== self.labels[attackConfig][i]:
                    indizeList.append(i)
                    refInputs.append(self.refInputs[attackConfig][i])                        
                    advInput.append(self.adversarialInputs[attackConfig][i])
                    
            #Zu 3.Kalkulieren der mittleren Distortion anhand DeepFool-Logik auf Teilmenge             
            self.data["x"].append(DeepFoolMeasure(refname=self.refname,ord=self.norm,
                                                     original=refInputs,
                                                     adversarials = advInput).get())

    def createPlot(self,size,x_step,y_step,x_max,y_max,useLowerBound=False,usePoinLabels=False ):
        self.plot = MultiPlot(x_label=self.xSuffix,y_label=self.ySuffix,
                              size=size,x_step=x_step,y_step=y_step,x_max=x_max,y_max=y_max,title=self.description)
        
        self.plot.addData(name=path.basename(self.refname) ,x=self.data["x"],y=self.data["y"],useLowerBound=useLowerBound,
                          usePoinLabels=usePoinLabels,pointlabels=self.data["names"])

class ScatterAccuracyQueryCalls(Plot):
    def __init__(self, refname, **kwargs):
        super().__init__(refname, **kwargs)
        self.name = "AkkuranzByCalls" 
        self.description = "Robuste Akkuranz in Abhängigkeit der QueryAnzahl " 

    def _calculate(self):
        kwX = {}
        self.data["x"] = MeandAdvQueryCounts(refname=self.refname, kwargs=kwX).get()

        kwY= {"output":self.rawY}
        self.data["y"]=Accuracy(refname=self.refname,kwargs=kwY).get()
    
class Bundle:
    def __init__(self,name,idlist:list) -> None:
        
        self.paths = []
        for identifier in idlist:
            try:
                aP = AttackProtocol(identifier)
                aP.load()
                self.paths.append(aP.getPath())
            except:
                FileNotFoundError("No protocol with ID ",identifier,"available.")
        
        self.folder = path.join("data","interpretation","Compare",name)
        self.name = name

    def getPaths(self):
        return self.paths

    def getName(self):
        return self.name

    def save(self):
        subFolder,subsubFolder = path.split(self.folder)
        mainFolder,subFolder = path.split(subFolder)
        pH = PathHandler(mainFolder)
        pH.create_subdirectory(subFolder,subsubFolder)
        filename = path.join(self.folder,"ID_bundle.json")
        
        data = {"name":self.name,"counts":len(self.paths),"paths":self.paths}
        jH = JSONHandler()
        jH.setData(data)
        jH.save(filename)

    def getFolder(self):
        return self.folder

class AccuracyPerturbationComp(Interpreter):
    """Berechnung von Accuracy/Distortion zu einer Kombination von Angriffsausführung(en)"""
    def __init__(self, refname, **kwargs):
        super().__init__(refname, **kwargs)
        try:
            self.data = kwargs["data"] 
        except :
            raise ValueError("Parameter 'data' is missing.")
        try:
            self.norm = kwargs["norm"] 
        except :
            raise ValueError("Parameter 'norm' is missing.")
        self.name = "AccuracyPerturbationCurve_L"+str(self.norm)+"_VGL"
        self.description = "Robuste Akkuranz in Abhängigkeit der "+str(self.norm)+"Metrik"

        #self._checkConsitency()

        for attack,value in self.data.items():
            interpret = AccuracyPerturbationCurve(self.refname,data=value,norm = self.norm)
            data = interpret.get()
            self.data[attack] = (data["x"], data["y"])
        
        self.plot = None
        
    def get(self):
        return self.data
    
    def createPlot(self,size,x_step,y_step,x_max,y_max):
        self.plot = MultiPlot(x_label="Mittlere L"+str(self.norm)+" Störung",y_label="Robuste Akkuranz",
                              size=size,x_step=x_step,y_step=y_step,x_max=x_max,y_max=y_max,title=self.description)
        data = self.get()
        for key,data in data.items(): 
            self.plot.addData(name=path.basename(key) ,x=data[0],y=data[1],useLowerBound=True)

    def loadData(self, folder):
        # Check ob Bundle-Ordner
        jH = JSONHandler()
        filename = path.join(folder,"ID_bundle.json")
        jH.load(filename)

        self.paths = jH.getData()["paths"]
        self._checkConsitency()

    def _checkConsitency(self):
        dataset = []
        config =[]
        attack=[]

        for p in self.paths:
            ap = AttackProtocol(path.basename(p))
            apDs = ap.getDataset()
            dataset.append(apDs["name"]+str(apDs["size"])+str(apDs["shape"]))
            attack.append(ap.getAName())
            params = []
            runs = ap.getRTData()
            for _, value in runs.items():
                params.append(value["parameter"]) # folder und runtime dürfen nicht vgl werden
            config.append(params)
        # Ensure same attack-configs for comparison
            
        datasetSet = set(dataset)
        if len(datasetSet)> 1: 
            raise ValueError("Consistency-check failed. Referenced attacks based on", len(datasetSet)," different datasets. Comparison not allowed.") 
        
        attackSet = set(attack)
        if len(attackSet)> 1:
            raise ValueError("Consistency-check failed. Referenced attacks based on", len(attackSet)," different attacktyps. Comparison not allowed.")         
    
    def show(self):
        self.plot.show()

    def save(self, folder ="./data/interpretation"):
        pH = PathHandler(folder)
        subdir = pH.create_subdirectory(self.refname)
        filename_figure = path.join(subdir,self.name+".jpg")
        filename_data = path.join(subdir,self.name+".json")
        
        self.plot.save(filename_figure) 
        jH = JSONHandler()
        savingData = dict()
        data = self.get()
        for key,(x,y) in data.items():
            savingData[key]=(str(x),str(y))
        jH.setData(savingData)
        jH.save(filename_data)
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
        print("hallo")
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

class ImagePerturbation(Interpreter):

    def __init__(self, refname, **kwargs):
        super().__init__(refname, **kwargs)
        self.name = "Perturbation" 
        self.description = "Perturbation" 
        self.original = dict()
        self.adversarial = dict()
        self.data = dict()

        try:
            for attackPath,attackdata in kwargs["data"].items() : 
                self.original[attackPath] = attackdata["0_1"]["input"]["ref"] #Vorerst nur das erste ref Bild
                self.adversarial[attackPath] = attackdata["0_1"]["input"]["advList"][0] #Vorerst nur das erste adv Bild
        except :
            raise ValueError("Parameter 'data' is missing.")


        for key in self.original:
            tensorlist = [Tensor(np.transpose(self.original[key], (2, 0, 1)).astype(np.float32)),
                          Tensor(np.transpose(np.subtract(self.original[key],self.adversarial[key]), (2, 0, 1)).astype(np.float32)),
                          Tensor(np.transpose(self.adversarial[key], (2, 0, 1)).astype(np.float32)),]

            self.data[key]= utils.make_grid(tensor=tensorlist,nrow=3)

    def get(self):
        return self.data

    def save(self, folder="./data/interpretation"):
        for key,value in self.data.items():
            ph = PathHandler(folder)
            subdir = ph.create_subdirectory(self.refname)
            filename = path.join(subdir,self.name+path.basename(key)+"_.png")
            utils.save_image(tensor=value,fp=path.join(filename))
"""
class AccuracyByL2VGL(YByXVGL):
    Berechnung von Accuracy/Distortion zu einer Kombination von Angriffsausführung(en)
    def __init__(self):
        super().__init__()
        self.name = "AkkuranzByL2VGL"
        self.description = "Robuste Akkuranz in Abhängigkeit der euklischen Metrik"
        # self._initSuffix(x="L2",y="Robuste Akkuranz")
        # self._initInterpreter(AccuracyByL2)
class AccuracyByLinfVGL(YByXVGL):
    def __init__(self):
        super().__init__()
        self.name = "AkkuranzByLinfVGL"
        self.description = "Robuste Akkuranz in Abhängigkeit der Maximums-Metrik"
        #self._initSuffix(x="Linf",y="Robuste Akkuranz")
        #self._initInterpreter(AccuracyByLInf)
class AccuracyByCallsVGL(YByXVGL):
    Berechnung von Accuracy/Distortion zu einer Kombination von Angriffsausführung(en)
    def __init__(self):
        super().__init__()
        self.name = "AkkuranzByCallsVGL"
        self.description = "Robuste Akkuranz in Abhängigkeit der Query-Anzahl"
        self._initSuffix(x="Calls",y="Robuste Akkuranz")
        self._initInterpreter(AccuracyByCalls)
"""