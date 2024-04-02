from classes.framework import Interpreter, JSONHandler, AttackProtocol, PathHandler, MultiPlot, AttackDataLoader
from os import path
import numpy as np
from torch import Tensor
from torchvision import utils
from matplotlib import pyplot as plt
from scipy.interpolate import BSpline

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
        self.name = "Accuracy" 
        self.description = "Accuracy" 
        try:
            self.labels = kwargs["labels"] 
        except :
            raise ValueError("Parameter 'labels' is missing.")
        try:
            self.predictions = kwargs["predictions"] 
        except :
            raise ValueError("Parameter 'predictions' is missing.")
        try:
            self.type= kwargs["type"] 
        except :
            raise ValueError("Parameter 'type' is missing.")

        self._calculate()

    def _calculate(self):
        total=0
        correct=0
        for i in range(len(self.predictions)):

            total += 1
            if self.predictions[i] == self.labels[i]:
                correct +=1
        self.data = correct/total

    def get(self):
        return self.data

    def save(self, folder="./data/interpretation"):
        jh = JSONHandler()
        filenameabs = path.join(folder,self.refname,self.type+"_"+self.name+".json")
        j = dict()
        j[self.refname]=self.data

        jh.setData(j)
        jh.save(filename=filenameabs)
   
class DistortionLp(Interpreter):

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
        self.description = "L"+str(self.ord)+" Metric"

        self._calculate()

    def get(self):
        return self.data

    def _calculate(self):
        dist = []
        for i in range(len(self.referencesData)):
            p = getPerturbation(self.adversarialData[i], self.referencesData[i])
            dist.append(np.linalg.norm(x=p, ord=self.ord))#axis=1

        if len(dist) == 0:
            dist =[0]

        self.data = dist
 
    def save(self, folder="./data/interpretation"):
        jh = JSONHandler()
        filenameabs = path.join(folder,self.refname,self.name+".json")
        j = dict()
        j[self.refname]=self.data

        jh.setData(j)
        jh.save(filename=filenameabs)

class DeepFoolMeasure(DistortionLp):
    def _calculate(self):
        dist = []
        #Calc Distortion by superclass
        s = super()
        s._calculate()
        #For all elements calculated by superclass divide datasize
        for i, el in enumerate(self.data):
            dist.append(el/np.linalg.norm(x=np.reshape(self.referencesData[i],[-1]),ord=self.ord))
        self.data = dist

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
        pass

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
            #Calc Distortion 
            dists = []
            advInput = np.array(self.adversarialInputs[key])
            refInput = np.array(self.refInputs[key])
            labels = np.array(self.labels[key])
            refPredictions = np.array(self.refPredictions[key])
            dists = np.array(DistortionLp(refname=self.refname,original=refInput,adversarials=advInput, ord= self.norm).get())
            dists[refPredictions != labels] = 0 #Just distortions where classificaton of referencedata = groud truth label  
            dists.sort(axis=0)

            #Calc y 
            probs = 1/float(len(self.refPredictions[key])) * np.arange(1, len(self.refPredictions[key])+1)
            probs[np.isnan(dists)] = 1.0
            dists = np.nan_to_num(dists, nan = np.nanmax(dists))

            self.data[key]={"x" : dists, "y": probs }

    def createPlot(self,size,x_step,y_step,x_max,y_max,useLowerBound=False,usePoinLabels=False ):
        self.plot = MultiPlot(x_label=self.xSuffix,y_label=self.ySuffix,
                              size=size,x_step=x_step,y_step=y_step,x_max=x_max,y_max=y_max,title=self.description)
        for key,data in self.data.items(): 
            self.plot.addCurve(name=path.basename(key) ,x=data["x"],y=data["y"])

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

        self.xSuffix = "L"+str(self.norm)+"-Metric"
        self.ySuffix = "Robust Accuracy"
        self._calculate()

    def _calculate(self):
        self.data["x"] = []
        self.data["y"] = []
        self.data["names"] = []
        self.data["lbX"] = []
        self.data["lbY"] = []
        
        for attackConfig in self.refInputs:
            #TBD: Check if only advExample should be considered for calculation
            advExampleIdxs = np.array(self.adversarialPredictions[attackConfig])!=np.array(self.refPredictions[attackConfig])
            trueAdvInputs = np.array(self.adversarialInputs[attackConfig])
            trueAdvInputs = trueAdvInputs[advExampleIdxs]

            #1. Calc robust accuracy for all Instances created by attack (acvInput - corresponding output)
            self.data["names"].append(path.basename(attackConfig))
            self.data["y"].append(Accuracy(refname=self.refname,labels= self.labels[attackConfig],
                                    predictions=self.adversarialPredictions[attackConfig],type="AbsoluteRobust").get())

            #2. Calc Distortion between originalData and adversarial Input for all Instances 
            dist = DistortionLp(refname=self.refname,ord=self.norm,
                                                     original=self.refInputs[attackConfig],
                                                     adversarials = self.adversarialInputs[attackConfig]).get()

            #3. Get mean of distortions for instances originalData missclassified by the classifier to 0             
            dist = np.array(dist)

            dist[np.array(self.refPredictions[attackConfig]) != np.array(self.labels[attackConfig])] = 0 #Just distortions where classificaton of referencedata = groud truth label  
            dist = np.ma.masked_equal(dist,0) # mask null-values to make sure they are not taken into account for mean-calculation 
            self.data["x"].append(np.mean(dist))

        self.data["lbX"],self.data["lbY"]=self.getLowerBound()

    def createPlot(self,size,x_step,y_step,x_max,y_max,useLowerBound=False,usePoinLabels=False ):
        self.plot = MultiPlot(x_label=self.xSuffix,y_label=self.ySuffix,
                              size=size,x_step=x_step,y_step=y_step,x_max=x_max,y_max=y_max,title=self.description)
        
        self.plot.addScatteredData(name=path.basename(self.refname) ,x=self.data["x"],y=self.data["y"],useLowerBound=False,
                          usePoinLabels=usePoinLabels,pointlabels=self.data["names"])
        
        self.plot.addCurve(path.basename(self.refname),x=self.data["lbX"], y=self.data["lbY"])


    def getLowerBound(self):
        # get sorted accuracy-indexlist
        rng = range(len(self.data["y"]))
        accIndPoints = [(i,self.data["y"][i]) for i in rng] # indize, accuracy
        accIndPoints.sort(key=lambda x: x[1])  
        accIdxsorted = [i for i,_ in accIndPoints]

        idx = accIdxsorted[:2] # initialize indexarray
        for i in accIdxsorted[2:]:
            if self.data["x"][i] < self.data["x"][idx[0]]:
                idx.insert(0,i)
            if self.data["x"][i]>self.data["x"][idx[-1]]:
                idx.append(i)
        
        x_curve = []
        y_curve = []

        for i in idx:
            x_curve.append(self.data["x"][i])
            y_curve.append(self.data["y"][i])
        return x_curve, y_curve

class AccuracyPerturbationBudget(Plot):
    """Berechnung von Accuracy/Distortion zu einer (Kombination von) Angriffsausführung(en)"""
    def __init__(self, refname, **kwargs):
        super().__init__(refname, **kwargs)
        try:
            self.norm = kwargs['norm'] 
        except :
            raise ValueError("Parameter 'norm' is missing.")
        
        self.name = "AccuracyPertubationBudget"+str(self.norm)
        self.description = "Accuracy vs. Perturbationbudget"  

        self.xSuffix = "L"+str(self.norm)+"-Metric"
        self.ySuffix = "Robust Accuracy"
        self._calculate()

    def _calculate(self):
        for key in self.adversarialInputs:
            self.data[key]=self.addData(inputAdv=self.adversarialInputs[key], 
                        inputRef= self.refInputs[key], 
                        labels=self.labels[key], 
                        outputAdv=self.adversarialPredictions[key], 
                        outputRef=self.refPredictions[key])
            
    def addData(self,inputRef:list,inputAdv:list,outputRef:list,outputAdv,labels:list):
        data = {}
        data["x"] = []
        data["y"] = []
        data["x"].append(0)#x=Distortion
        acc = Accuracy(refname="Test",labels=labels, predictions=outputRef, type="CleanRobust")                
        data["y"].append(acc.get())#y=Accuracy

        #Calc Distortion between originalData and adversarial Input for all Instances 
        distortion = DistortionLp(refname="C_"+self.refname,ord=self.norm,
                                 original=inputRef,
                                 adversarials = inputAdv).get()        
        
        #TBD: Filtere nach adv. BSP
        distArray = np.array(distortion)
        #sortiere advPredition, label und refPrediction anhand Distortion


        #All unpurturbated points are contained in the referencedata D_clean (inputdata for attack)
        #So at x=0 set y = Accuracy(D_clean)
        
        points = []
        d=[]
        o=[]
        l=[]
        for index in range(len(outputRef)):
            d.append(0)
            o.append(outputRef[index])
            l.append(labels[index])
        points.append((0,o,l))
        points+=[(distortion[i],[outputAdv[i]],[labels[i]]) for i in range(len(distortion))]
        points.sort(key=lambda x: x[0])

        #Falls ein Distortion mehrfach vorliegt, bilde mittelwert
        duplicates = {}
        for i, point in enumerate(points):
            if point[0] in duplicates:
                duplicates[point[0]].append(i)
            else:
                duplicates[point[0]] = [i]

        j=1
        for i in range(1,len(points)):
            pred = []
            lbl = []

            if j==i:
                if len(duplicates[points[i][0]])>1:
                    j+=len(duplicates[points[i][0]])
                    for k in range(j):
                        pred+=(points[k][1])
                        lbl+=(points[k][2])  
                else:

                    for k in range(i):
                        pred+=points[k][1]
                        lbl+=points[k][2]  
                    j+=1

                acc = Accuracy(refname="Test",labels=lbl, predictions=pred, type="AbsRobust")                
                data["x"].append(points[i][0])#x=Distortion
                data["y"].append(acc.get())#y=Accuracy

        #points+=[(distortion[i],o+outputAdv[:i+1],l+labels[:i+1]) for i in range(len(distortion))]

        return data
   

    def createPlot(self,size,x_step,y_step,x_max,y_max,useLowerBound=False,usePoinLabels=False ):
        self.plot = MultiPlot(x_label=self.xSuffix,y_label=self.ySuffix,
                              size=size,x_step=x_step,y_step=y_step,x_max=x_max,y_max=y_max,title=self.description)
        #self.plot.addvLines(x=self.data["names"])
        
        
        for key in self.data:
            #xnew = np.linspace(min(self.data[key]["x"]), max(self.data[key]["x"]), 600)  
            #smooth = BSpline(xnew,self.data[key]["x"], self.data[key]["y"])
            

            self.plot.addCurve(name=path.basename(self.refname) ,x=self.data[key]["x"],
                               y=self.data[key]["y"])

class AccuracyPerturbationBudgetAggregate(AccuracyPerturbationBudget):
    def __init__(self, refname, **kwargs):
        super().__init__(refname, **kwargs)
        self.name ="AggregationOfAccuracyPerturbationCurves"
    def _calculate(self):
        commonAdversarialInputs = []        
        commonRefInputs = []
        commonRefPredictions = []
        commonAdversarialPredictions = []
        commomLabels = []
        for key in self.adversarialInputs:
            commonAdversarialInputs+= self.adversarialInputs[key]
            commonRefInputs += self.refInputs[key]
            commonAdversarialPredictions += self.adversarialPredictions[key]
            commonRefPredictions += self.refPredictions[key]
            commomLabels += self.labels[key]
        self.data[self.refname] =self.addData(inputAdv=commonAdversarialInputs, 
                        inputRef= commonRefInputs, 
                        labels=commomLabels, 
                        outputAdv=commonAdversarialPredictions, 
                        outputRef=commonRefPredictions)
class ScatterAccuracyQueryCalls(Plot):
    def __init__(self, refname, **kwargs):
        super().__init__(refname, **kwargs)
        self.name = "AccuracyByCalls" 
        self.description = "Robust accuracy by querynumber " 

    def _calculate(self):
        kwX = {}
        self.data["x"] = MeandAdvQueryCounts(refname=self.refname, kwargs=kwX).get()

        kwY= {"output":self.rawY}
        self.data["y"]=Accuracy(refname=self.refname,kwargs=kwY,type="AbsoluteRobust").get()
    
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
        self.description = "Robust accuracy by L "+str(self.norm)+" metric"
        #self._checkConsitency()

        for attack,value in self.data.items():
            interpret = AccuracyPerturbationCurve(self.refname,data=value,norm = self.norm)
            data = interpret.get()
            self.data[attack] = (data["x"], data["y"],data["lbX"],data["lbY"])
        
        self.plot = None
        
    def get(self):
        return self.data
    
    def createPlot(self,size,x_step,y_step,x_max,y_max):
        self.plot = MultiPlot(x_label="L"+str(self.norm)+" perturbationbudget",y_label="Robust accuracy",
                              size=size,x_step=x_step,y_step=y_step,x_max=x_max,y_max=y_max,title=self.description)
        data = self.get()
        for key,value in data.items(): 
            self.plot.addScatteredData(name=path.basename(key) ,x=value[0],y=value[1],useLowerBound=True)
            self.plot.addCurve(name=path.basename(key),x = value[2],y=value[3])

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
        for key,(x,y,lbx,lby) in data.items():
            savingData[key]=(str(x),str(y),str(lbx),str(lby))
        jH.setData(savingData)
        jH.save(filename_data)



class AccuracyPerturbationBudgetComp(Plot):#AccuracyPerturbationComp):
    """Berechnung von Accuracy/Distortion zu einer Kombination von Angriffsausführung(en)"""
    def __init__(self, refname, **kwargs):
       # super().__init__(refname, **kwargs)
        try:
            self.i = kwargs["data"] 
        except :
            raise ValueError("Parameter 'data' is missing.")
        try:
            self.norm = kwargs["norm"] 
        except :
            raise ValueError("Parameter 'norm' is missing.")
        self.name = "AccuracyPerturbationBudget_L"+str(self.norm)+"_VGL"
        self.description = "HopSkipJump"#"Robust accuracy by L "+str(self.norm)+" metric"
        self.data = {}
        #self._checkConsitency()
        self.refname = refname
        for attack,value in self.i.items():
            #interpret = AccuracyPerturbationBudgetAggregate(self.refname,data=value,norm = self.norm)
            interpret =AccuracyPerturbationBudget(refname=path.basename(attack),data=value, 
                                                  norm=self.norm )
            data = interpret.get()
            for j,attackConst in enumerate(data): #Erstml nur eerste attacke
                if j == 0:
                    self.data[attack] = (data[attackConst]["x"], data[attackConst]["y"])

        self.plot = None
        
    
    def createPlot(self,size,x_step,y_step,x_max,y_max):
        self.plot = MultiPlot(x_label="L"+str(self.norm)+" Perturbation",y_label="Robust accuracy",
                              size=size,x_step=x_step,y_step=y_step,x_max=x_max,y_max=y_max,title=self.description)
        data = self.get()
        for key,value in data.items(): 
            t = key.replace("HSJ","")
            if t == "":
                t="Baseline"
            self.plot.addCurve(name=t,x = value[0],y=value[1])
            #self.plot.addCurve(name=path.basename(key),x = value[0],y=value[1])


class RobustnessCurveComp(Plot):#AccuracyPerturbationComp):
    """Berechnung von Accuracy/Distortion zu einer Kombination von Angriffsausführung(en)"""
    def __init__(self, refname, **kwargs):
       # super().__init__(refname, **kwargs)
        try:
            self.i = kwargs["data"] 
        except :
            raise ValueError("Parameter 'data' is missing.")
        try:
            self.norm = kwargs["norm"] 
        except :
            raise ValueError("Parameter 'norm' is missing.")
        self.name = "RobustnessCurve"+str(self.norm)+"VGL"
        self.description = "GeoDa"#"Robust accuracy by L "+str(self.norm)+" metric"
        self.data = {}
        #self._checkConsitency()
        self.refname = refname
        for attack,value in self.i.items():
            #interpret = AccuracyPerturbationBudgetAggregate(self.refname,data=value,norm = self.norm)
            interpret =RobustnessCurve(refname=path.basename(attack),data=value, 
                                                  norm=self.norm )
            data = interpret.get()
            for j,attackConst in enumerate(data): #Erstml nur eerste attacke
                if j == 0:
                    self.data[attack] = (data[attackConst]["x"], data[attackConst]["y"])

        self.plot = None
        
    
    def createPlot(self,size,x_step,y_step,x_max,y_max):
        self.plot = MultiPlot(x_label="L"+str(self.norm)+" Perturbation",y_label="Robust accuracy",
                              size=size,x_step=x_step,y_step=y_step,x_max=x_max,y_max=y_max,title=self.description)
        data = self.get()
        for key,value in data.items(): 
            t = key.replace("GeoDa","")
            if t == "":
                t="Baseline"
            self.plot.addCurve(name=t,x = value[0],y=value[1])
            #self.plot.addCurve(name=path.basename(key),x = value[0],y=value[1])




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

def getPerturbation(referenceData, adversarialData):
    return np.reshape(np.subtract(adversarialData, referenceData), [-1])

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