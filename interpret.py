from classes.framework import DataHandler, Interpreter, RestApiCall, AttackProtocol, saveJSON ,PathHandler
from os import path
import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor
from torchvision import utils
from classes.data import ImageByModelQuery

def getAdvAccuracy(outputs):
    total=0
    correct=0
    for _,value in outputs.items():
        total += len(value)-1 #Erstes Element enthält stets wert 0, da es sich auf referenzbild bezieht
        correct += len([y for y in value[1:] if y == value[0]])
    return correct/total

class AccuracyByDistortion(Interpreter):
    """Berechnung von Accuracy/Distortion zu einer (Kombination von) Angriffsausführung(en)"""
    def __init__(self, id):
        #Lade Protokoll
        ad = AttackProtocol(id)
        self.filename =path.join(ad.getPath(),"Accuracy_Perturbation_Curve.jpg")

        self.outputs = ad.getOutputs()
        self.inputs =  ad.getInputs()

        #Lese zugehörigen output
        self.distortion = dict()
        self.accuracy = dict()
        self.points = list()

    def calculate(self):    
        
        #TBD : Logik entsprechend https://ieeexplore.ieee.org/document/9413143/
        if self.inputs.keys() != self.outputs.keys():
            ValueError("Eingabe- und Ausgabedaten passen nich zusammen!")
        
        dist = []
        for attack in self.outputs:
        #Berechnung der Akkuranz
            self.accuracy[attack]=getAdvAccuracy(self.outputs[attack])
        
        #Berechnung der Distortion
            liste = []
            for nparrays in self.inputs[attack]: 
                x_ref = nparrays[0] #Referenzbild            
                perturbations = [np.reshape(x_adv - x_ref, [-1]) for x_adv in nparrays[1:]]
                for perturbation in perturbations:
                    dist.append(np.linalg.norm(x=perturbation)) #
                
            self.distortion[attack] = sum(dist)/len(dist)

        #Erstellung von Punkten für Kurve
            self.points.append((self.distortion[attack],self.accuracy[attack]))

        #Kurvenpunkte werden anhand des Distortionwertes aufsteigend sotriert
        self.points.sort(key=lambda x: x[0])  # index 1 means second element
        x_values = [x for (x,_) in self.points]
        y_values = [y for (_,y) in self.points]
        
        steps = 0.1

        x_label = "Perturbation-Budget"
        y_label = "Robust Accuracy"

        plt.figure(figsize=(5,5))
        plt.plot(x_values, y_values, "*-")
        plt.yticks(np.arange(0, 1.1, step=0.1)) # Values in [0,1] 
        plt.xticks(np.arange(0, x_values[-1]+steps, step=steps))
        plt.xlabel(xlabel=x_label)
        plt.ylabel(ylabel=y_label)
        
        plt.title(x_label+" vs. "+y_label)


        plt.savefig(self.filename+".jpg")
        

    def save(self):
        liste = []
        for p in self.distortion:
            liste.append({p:{"Distortion": self.distortion[p], "Accuracy":self.accuracy[p]}})
        saveJSON(filename=self.filename+".json",data=liste)
        
class AdvExamplesImages(Interpreter):
    def __init__(self, id):
        protocol = AttackProtocol(id)
        self.inputs = protocol.getInputs()
        self.savingPath = protocol.getPath()
        self.grids = dict()

    def calculate(self):
        for key,value in self.inputs.items():
            tensorlist = []
            for nparrays in value: 
                tensorlist.append(Tensor( np.transpose(nparrays[0], (2, 0, 1)).astype(np.float32))) #Referenzbild            
                tensorlist.append([Tensor( np.transpose(array, (2, 0, 1)).astype(np.float32)) for array in nparrays[1:]][0])#Adv.Bsp
            
            self.grids[key] =utils.make_grid(tensor=tensorlist,nrow=len(tensorlist)//len(value))#len(tensorlist)/len(self.input))

    def save(self):
        for key,grid in self.grids.items():
            utils.save_image(tensor=grid,fp=path.join(self.savingPath,"AD_"+path.basename(key)+".jpg"))
        
id = "ID03"
acc = AccuracyByDistortion(id)
acc.calculate()
acc.save()
test = AdvExamplesImages(id)
test.calculate()
test.save()


#dist = AccuracyByDistortion(data)
#dist.calculate()
#dist.save("./Test/t/Kimba5/")

"""
testdata = ["./Test/crawl/ID03"]
acc = AccuracyByDistortion()
acc
tes=loadOutput(testdata)
print(getAdvAccuracy(tes))
"""


class AccuracyByQueryCounts(Interpreter):
    """Berechnung von Accuracy/Distortion zu einer (Kombination von) Angriffsausführung(en)"""
    def __init__(self, id):
        #Lade Protokoll
        ad = AttackProtocol(id)
        self.filename =path.join(ad.getPath(),"Accuracy_Perturbation_Curve")

        self.outputs = ad.getOutputs()
        self.querycounts = "" # TBD

        #Lese zugehörigen output
        self.points = list()

    def calculate(self):    
        pass
class Runtime(Interpreter):
    pass
    #Gegenüberstellung der Laufzeiten
    #Laden mehrerer Protokolle (bspw. zu gleichen Angriff+ -sparametern für Vanilla und Verteidigungsmodell) 
    #Vergleich


class Transferability(Interpreter):

    def __init__(self,data):
        self.input = dict()
        self.output = dict()
        self.acc = dict()

        #for p in data:
         #   self.input[p] = loadInput(p)
        self.collector = ImageByModelQuery()

    def setRequestHandler(self,apiCall:RestApiCall): 
        self.collector.setRequestHandler(apiCall)

    def calculate(self):
        self.acc = dict()

        """for key in testdata:
            self.output[key] = self.collector.collectData(inputdata=self.input[key])
            self.acc[key]= getAdvAccuracy(self.output[key])#To Check hier ist ggf. garnic
        """
        return self.acc
    
    def save(self,folder):
        filename =path.join(folder,"T_"+self.ap)
        saveJSON(filename=filename,data=self.acc)
        

class TransferabilityMatrix(Interpreter):
    def __init__(self, data):
        #TBD:Transfer-Protokolle auslesen
        
        self.matrix = None

    def calculate(self):
        pass
    
    def show(self):
        pass
 
    # creating a DataFrame
        """dict = {'Name' : ['Martha', 'Tim', 'Rob', 'Georgia'],
                    'Maths' : [87, 91, 97, 95],
                    'Science' : [83, 99, 84, 76]}
            df = pd.DataFrame(dict)
            
            # displaying the DataFrame
            df.style"""






"""
class ImageDistortion:

    def __init__(self):
        self.l2 = [] # List of lists of distorsions between x_ref and x_adv
        self.linf = [] # List of lists of distorsions between x_ref and x_adv  
        
    def measurel2(self, x_ref, x_adv:list)->list:
        temp = []
        for x in x_adv:
            perturbation = np.reshape(x - x_ref, [-1])
            temp.append(np.linalg.norm(x=perturbation))
        self.l2.append(temp)
    
    def measurelinf(self,x_ref, x_adv:list)->list:
        temp = []
        for x in x_adv:
            perturbation = np.reshape(x - x_ref, [-1])
            temp.append(np.linalg.norm(x=perturbation,ord=np.inf))
        self.linf.append(temp)

    def saveData(self, folder):
        saveListOfListsToCSV(list=self.l2, filename=path.join(folder,"l2_Distortion.csv"))
        saveListOfListsToCSV(list=self.linf, filename=path.join(folder,"linf_Distortion.csv"))

def getAdvAccuracy(outputs:list, target=None,)->list:
    #1. Fall: untargeted
    correct = 0
    total = 0
    for o in outputs: #o[0] contains label of reference-input, o[1:] contains labels for adv. examples of reference-input
        total += len(o[1:])
        correct += len([y for y in o[1:] if y == o[0]])
    return correct/total


args = testHSJ()

dist = ImageDistortion()

ph = PathHandler(args.data_dir) # Angriffsausführung
subdirs = ph.get_subdirectories() #-> Ordner zur Param 

adv_accuracy = [] # List of Accuranz/epsilon
data = dict()
number = 0
correct = 0
total = 0

class DistortionI(Interpreter):
    
    def calculate(self):
        l2 = []
        linf = []
        for x in self.data:
            x_adv = "TBD"
            x_ref = "TBD"

            perturbation = np.reshape(x_adv - x_ref, [-1])
            #t2.append(np.linalg.norm() (x=perturbation))
            #tinf.append(np.linalg.norm(x=perturbation,ord=np.inf))

        #self.l2.append(t2)

        #Berechne dist. aller adv. Bsp. 

    def show(self):
        pass
        #TBD, ggf. Not implemented 
        
    def save(self):
        pass
    """

"""
#Lade Daten aus Verzeichnis
for dir in subdirs:
    dataH = DataHandler()
    dataH.load(dir)
    data[path.basename(dir)] = dataH.getData()
    total += len(data[path.basename(dir)])

    #Berechne Distortion
    dist.measurel2(data[dir][0],data[dir][1:])
    dist.measurelinf(data[dir][0],data[dir][1:])

dist.saveData(ph.path)
#Speichere Grid mit Originalbild + adv Example
rows = (total)//len(data)
l = [ data[key] for key in data]
saveListOfListsToGrid(nd_list=l, filename=path.join(ph.path,"grid.png"),rows=rows)


#Zeige daten in epsilon-abhängigkeit.
#TBD: Verwendung von epsilon-Werten in Angriffsauführung?!

#y_data = readOutput(path.join(args.data_dir,"output.csv"))
#acc= getAdvAccuracy(outputs=y_data)
#TBD: Abspeicherung von Accuranz zu Epsilon-Werten notwendig!


#Zeige daten in Aufrufabhängigkeit


"""