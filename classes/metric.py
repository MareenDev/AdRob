from classes.framework import Interpreter, JSONHandler
from os import path
import numpy as np

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
            self.output = kwargs["predictions"] 
        except :
            raise ValueError("Parameter 'predictions' is missing.")
        
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
            self.type= "Test"

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
            self.referenceData = kwargs["references"] 
        except :
            raise ValueError("Parameter 'references' is missing.")
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
        for i in range(len(self.referenceData)):
            p = getPerturbation(self.adversarialData[i], self.referenceData[i])
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
            dist.append(el/np.linalg.norm(x=np.reshape(self.referenceData[i],[-1]),ord=self.ord))
        self.data = dist

def getPerturbation(referenceData, adversarialData):
    return np.reshape(np.subtract(adversarialData, referenceData), [-1])
