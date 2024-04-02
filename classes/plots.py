from classes.framework import Interpreter, JSONHandler, AttackProtocol, PathHandler, MultiPlot, AttackDataLoader
from os import path
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import BSpline
from classes.metric import DistortionLp, Accuracy

def makeDataLists(data:dict):
    predictionsRef =  dict()
    predictionsAdv =  dict()
    inputsRef = dict()
    inputsAdv =dict()
    attackQueryCalls = dict()
    labels = dict()
    for key,attackData in data.items():#Loop über die verschiedenen Angriffsausführungen
        predictionsAdv[key] =[]
        predictionsRef[key] =[]
        inputsRef[key] =[]
        inputsAdv[key] =[]
        attackQueryCalls[key] = []
        labels[key] =[]

        for k,data in attackData.items(): #Loop über die Inputelement-Referenzen
            for i,advPred in enumerate(data["output"]["advList"]):#abhängigkeit zu x_ref fehlt
                predictionsAdv[key].append(advPred)    
                predictionsRef[key].append(data["output"]["ref"])
                inputsAdv[key].append(data["input"]["advList"][i])   
                inputsRef[key].append(data["input"]["ref"])
                labels[key].append(data["labels"])   
                #if i == 0: addieren der attackQueries notwendig bei mehreren 
                if i>=1:
                    attackQueryCalls[key].append(attackQueryCalls[key][i-1]+data["queryCalls"]["advList"][i])
                else:    
                    attackQueryCalls[key].append(data["queryCalls"]["advList"][i])   
    return labels, predictionsAdv, predictionsRef, inputsAdv, inputsRef, attackQueryCalls     

def filterListByIndizes(liste:list, indices:list):
    for i in range(liste):
        pass

class Plot(Interpreter):
    def __init__(self, refname, **kwargs):
        super().__init__(refname, **kwargs)


        labels, predictionsAdv,predictionsRef,adversarialInputs,refInputs,attackQueryCalls = makeDataLists(kwargs["data"])
        
        self.labels = labels
        self.predictionsAdv =  predictionsAdv
        self.predictionsRef =  predictionsRef
        self.inputsAdv = adversarialInputs
        self.inputsRef =refInputs
        self.attackQueryCalls =attackQueryCalls

        self.data = dict()
        self.plot = None
        self.xSuffix = "" 
        self.ySuffix = ""  
                    
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
        for key in self.inputsAdv: 
            #Calc Distortion 
            dists = []
            inputsAdv = np.array(self.inputsAdv[key])
            inputsRef = np.array(self.inputsRef[key])
            labels = np.array(self.labels[key])
            predictionsRef = np.array(self.predictionsRef[key])
            dists = np.array(DistortionLp(refname=self.refname,original=inputsRef,adversarials=inputsAdv, ord= self.norm).get())
            dists[predictionsRef != labels] = 0 #Just distortions where classificaton of referencedata = groud truth label  
            dists.sort(axis=0)

            #Calc y 
            probs = 1/float(len(self.predictionsRef[key])) * np.arange(1, len(self.predictionsRef[key])+1)
            probs[np.isnan(dists)] = 1.0
            dists = np.nan_to_num(dists, nan = np.nanmax(dists))

            self.data[key]={"x" : dists, "y": probs }

    def createPlot(self,size,x_step,y_step,x_max,y_max,useLowerBound=False,usePoinLabels=False ):
        self.plot = MultiPlot(x_label=self.xSuffix,y_label=self.ySuffix,
                              size=size,x_step=x_step,y_step=y_step,x_max=x_max,y_max=y_max,title=self.description)
        for key,data in self.data.items(): 
            self.plot.addCurve(name=path.basename(key) ,x=data["x"],y=data["y"])


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

        self.xSuffix = "L"+str(self.norm)+"-Metrik"
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
        
        
        for key in self.data:
            #xnew = np.linspace(min(self.data[key]["x"]), max(self.data[key]["x"]), 600)  
            #smooth = BSpline(xnew,self.data[key]["x"], self.data[key]["y"])
            

            self.plot.addCurve(name=path.basename(self.refname) ,x=self.data[key]["x"],
                               y=self.data[key]["y"])
