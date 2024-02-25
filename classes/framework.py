from os import path, mkdir, listdir
from torch import Tensor
from torchvision import utils

import numpy as np
import csv
import time
import json
import requests

class PathHandler:
    def __init__(self, directory):
        self.path = directory

    def get_directory(self):
        """
        Get attribute self.path
        Returns:
            corresponding directory.
        """
        return self.path

    def get_subdirectories(self):
        """
        List all subdirectories of a directory.

        Args:
            directory: The directory to list subdirectories from.

        Returns:
            A list of absolute paths of subdirectory.
        """
        subdirectories = []
        for entry in listdir(self.path):
            p = path.join(self.path, entry)
            if path.isdir(p):
                subdirectories.append(p)

        return subdirectories
    
    def create_subdirectory(self, *args):
        sub_folder = self.path
        for name in args:
            sub_folder = path.join(sub_folder,repl(name))
            if not path.exists(sub_folder):
                mkdir(sub_folder)
        
        return sub_folder

    def get_filenames(self, type)-> dict:
        """ List all filenames in a directory
        Args:
            type: filetype 
        Returns:
            A dictornary of filenames. key: folder-path, value: list of filenames in folder.
        """
        result = dict()

        dirs = self.get_subdirectories() + [self.get_directory()]
        for dir in dirs:
            filenames = []
            filenames += [each for each in listdir(dir) if each.endswith('.'+type)]
            if filenames != []:
                result[dir] = filenames

        return result


class DataHandler:
    def __init__(self):
        self.data = dict() 
        self.name = ""
        self.size = 0
        self.shape = tuple()

    def getName(self)->str:
        return self.name

    def setName(self,name):
        self.name = name

    def setData(self,data:dict):         
        self.data = data 
        
    def load(self,directory):
        filename = path.join(directory,"dataset.json")
        dataset = loadJson(filename)
        self.name = dataset["name"]
        self.shape = dataset["shape"]
        self.size = dataset["size"]
        data = dict()

        for filename in dataset["data"]: 
            data[filename]= np.load(filename) #Wichtig: x_ref daten müssen an Position 0 steht!!!        
        self.setData(data=data)
    
    def getShape(self):
        return self.shape

    def setShape(self,shape):
        self.shape=shape

    def getSize(self):       
        return self.size
    
    def setSize(self,size):       
        self.size=size

    def getData(self):
        return self.data

    def saveData(self, directory):
        for key in self.data:
            filename = path.join(directory,str(path.basename(key)))
            np.save(arr=self.data[key],file=filename)


class AttackProtocol:
    def __init__(self,id):    
        self.filename = path.join("data","attacks",id,"protocol_"+id+".json")
        self.start    = time.time()
        self.end      = None

        self.id         = id
        if path.exists(self.filename):
            self.load()
        else:
            self.aName      = ""
            self.reqHandlerClass = ""
            self.reqestUrl  = ""
            self.reqMethod  = ""
            self.dataset    = dict()
            self.RTData     = dict()
            self.counter    = 0

    def getID(self):
        return self.id        

    def getPath(self):
        return path.dirname(p=self.filename)

    def setAttackName(self,name):
        self.aName      = name

    def setApiData(self,handlerClass,url,method):
        self.reqHandlerClass = handlerClass
        self.reqestUrl  = url
        self.reqMethod  = method
        
    def setDataset(self,name,size,shape):
        self.dataset = { "name": name, "size":size, "shape":shape}

    def addRTData(self,folder,parameter, runtime):
        self.RTData[str(self.counter)] = {"folder":folder,
                                        "parameter":parameter,
                                        "runtime":self._getDurationFormatted(runtime)}
        self.counter +=1

    def stop(self):
        self.end = time.time()

    def getStart(self):
        return self.start
    
    def _getDurationFormatted(self,seconds):
        return str(int(seconds // 3600))+"h "+str(int((seconds % 3600)//60))+"min "+str(int(seconds % 60))+"sek"

    def save(self):
        #Check if Protocol already exist
        data = {"ID":self.id,
                "Attack":self.aName,
                "API":{
                    "Handler":self.reqHandlerClass,
                    "URL":self.reqestUrl,
                    "Method":self.reqMethod
                    },
                "Dataset":self.dataset,
                "Runs":self.RTData}        
        saveJSON(data=data,filename=self.filename)

    def load(self):
        data = loadJson(self.filename)
        self.aName = data["Attack"]
        self.counter    = len(data["Runs"])
        self.id         = data["ID"]
        self.reqHandlerClass = data["API"]["Handler"]
        self.reqestUrl  = data["API"]["URL"]
        self.reqMethod  = data["API"]["Method"]
        self.dataset    = data["Dataset"]
        self.RTData     = data["Runs"]

    def getOutputs(self):
        outputDict = dict()
        for _,value in self.RTData.items():
            output = self._loadOutput(value["folder"])
            outputDict[value["folder"]] =output
        return outputDict

    def getInputs(self):
        inputDict = dict()
        for _,value in self.RTData.items():
            input = self._loadInput(value["folder"])
            inputDict[value["folder"]] = input
        return inputDict

    def _loadOutput(self,p):
        filename = path.join(p,"Output.json")
        return loadJson(filename)

    def _loadInput(self,p):
        ph = PathHandler(path.join(p,"Input"))
        filenames = ph.get_filenames("npy")
        return [np.load(path.join(folder,filename)) for folder,filenameList in filenames.items() for filename in filenameList ] #[np.load("Test/Attack/ID03/SignOpt1708123971_9652371/Input/Bag_0.npy"),np.load("Test/Attack/ID03/SignOpt1708123971_9652371/Input/Bag_1.npy"),np.load("Test/Attack/ID03/SignOpt1708123971_9652371/Input/Bag_2.npy"),np.load("Test/Attack/ID03/SignOpt1708123971_9652371/Input/Bag_3.npy"),np.load("Test/Attack/ID03/SignOpt1708123971_9652371/Input/Bag_4.npy"),np.load("Test/Attack/ID03/SignOpt1708123971_9652371/Input/Bag_5.npy"),np.load("Test/Attack/ID03/SignOpt1708123971_9652371/Input/Dress_0.npy"),np.load("Test/Attack/ID03/SignOpt1708123971_9652371/Input/Dress_1.npy"),np.load("Test/Attack/ID03/SignOpt1708123971_9652371/Input/Dress_0.npy")]
        

#---------------- classes to inherent from for extention----------------

class Collector:
    """ Abstract class for Domain-specific data-handling"""
        
    def collectData(self)->dict:
        """ Collect begnin examples for classifier
        Args:
            number: Number of expected examples per label to collect.
            labels: list of labels to search data for
        """ 
        raise NotImplementedError
    
    def saveData(self, data:dict,folder)->None:
        """ Create tensors for data in directory/subdirectories """
        raise NotImplementedError


class DataPreparation:
    def __init__(self) -> None:
        self.shape = tuple()
        self.data = dict()
        self.size  = 0

    def load(self, directory):
        raise NotImplementedError()

    def prepare(self, directory, shape)->dict:
        raise NotImplementedError()

    def createDataset(self,directory,data:dict):
        """ 
            Save numpyarrays for images in the 
            directory self.ph.path and corresponding subdirectories
            Shape per numpyarray is (Height,Width,Channel) 
            mit Werten im Bereich [0,1]
        """   
        ph = PathHandler(directory)    
        subdir = ph.create_subdirectory("Dataset")
        dataset = {"name":directory, "size":self.size,"shape":self.shape}
        filenamelist = []

        for key in data:
            filename = path.join(subdir,path.basename(key)+".npy")    
            np.save(file=filename,arr=data[key])

            filenamelist.append(filename)
        
        dataset["data"] = filenamelist
        saveJSON(filename=path.join(subdir,"dataset.json"),data=dataset)
            

class RestApiCall:
    """ class to inherent from to query api-classifiers"""
    def __init__(self , url , method='POST', id = "C") :
        self.url = url
        self.method = method
        self.id = id
        self.resetCounts()

    def responseIsOk(self, res):
        """Checking if response is as expected
        Args: res - response to handle
        Return: True if response is as expected, False otherwise
        """
        result = False
        try:
            if res.status_code == 200:
                result = True
        except:
            raise ValueError("Invalide statuscode.",res.status_code)
        return result

    def _sendRequest(self, data):
        """Send Request corresponding to self.method
        Args: data - inputdata to construct API-request from 
        Return: response from API-call
        """
        try:
            header = self.createRequestHeader()

            if self.method == 'POST':
                payload = self.createRequestPayload( data= data )
                result = requests.post(url=self.url, headers= header, data=payload)
            elif self.method == 'GET':
                params = self.createRequestParameters(data=data)
                result = requests.get(url=self.url,params =params, headers=header)
            else:
                raise NotImplementedError("No handling for processing method",self.method,"implemented")
            return result
        except:
            raise ValueError("Error in request-processing")
        
    def getCounts(self):
        """Get counter
        Return: self.counter"""
        return self.counter#self.id+"-"+str(self.counter)
    
    def resetCounts(self):
        """Reset counter"""
        self.counter = 0

    def decrementCounts(self):
        """Decrement counter"""
        self.counter = self.counter -1 

    def predict(self, x):
        """Get prediction for input  
        Args: x - input data (numpyarray) to get prediction for
        Return:  numpyarray containing response from API-call"""
        try: 
            data = self._transformInput(x)
            resp = self._sendRequest(data)
            result = self._getDataByResponse(resp)
            self.counter +=1
            return result
        except:
            raise ValueError("Fehler bei der Verarbeitung der Daten dar Form",x.shape)
        
    def createRequestPayload(self, data):
        raise NotImplementedError("To be implemented in inherented class")
    
    def createRequestParameters(self, data):
        raise NotImplementedError("To be implemented in inherented class")
    
    def createRequestHeader(self):
        raise NotImplementedError("To be implemented in inherented class")
    
    def _getDataByResponse(self, resp):
        raise NotImplementedError("To be implemented in inherented class")      
    
    def _transformInput(self,x):
        """Transform input-data for further processing"""
        raise NotImplementedError("To be implemented in inherented class")


class EvasionAttack:
    def __init__(self,name,apiCall:RestApiCall,shape,targeted,norm,batchsize,verbose):
        self.initLists()
        self.protocolname = name
        self.attackname = ""
        self.APICall = apiCall
        self.shape = shape,
        self.targeted = targeted
        self.norm = norm,
        self.batchsize = batchsize
        self.verbose = verbose

    def initLists(self):
        self.x_list = dict()    # Dictonary of lists of input-data.  
        self.y_list = dict()        # Dictonary of lists of classifier output-data (classes).  
        self.querycount = dict()    # Dictonary of lists of query-counts - the sum of elements per sublist contains the total counts of API-calls for generating the last adv. ex

    def generate(self,x_ref,name,**kwargs):
        raise NotImplementedError()

    def _saveModelOutput(self,folder):
        #saveListOfListsToCSV(list=self.y_list, filename=path.join(folder,"output.csv"))
        pass

    def saveData(self,folder):
        if self.x_list != {}: #[]
            #Prepare folders for saving data
            pathH = PathHandler(folder)
            tmp = str(time.time()).replace(".","_")
            inputfolder = pathH.create_subdirectory(self.protocolname, self.attackname+tmp,"Input")
            attackfolder = path.dirname(inputfolder)
            #Save input-data

            dh = DataHandler()      #Idee:Adv Bilddaten als Dataset speichern
            dh.setName("adversarialExamples")      #TBD 
            dh.setSize(0)           #TBD
            dh.setShape(self.shape)  #TBD
            d = dict()
            for key, value in self.x_list.items(): 
                d[key]=value
#                dh.setData(data={key:value}) #Wichtig: x_ref daten müssen an 'Position 0' steht!!!      
            dh.setData(d)
            dh.saveData(inputfolder)  

            #Save model-output
            y_dict = dict()
            o_dict = dict()
            for key in self.y_list:
                y_dict[key] = [str(x) for x in self.y_list[key]] 
                o_dict[key] = [str(x) for x in self.querycount[key]]

            saveJSON(data=y_dict,filename=path.join(attackfolder,"Output.json"))
            saveJSON(data=o_dict,filename=path.join(attackfolder,"APICalls.json"))

        else: 
            attackfolder = None
        return attackfolder
    

class Interpreter:
    def __init__(self,data):
        self.data = data

    def calculate(self):
        NotImplementedError()
        
    def save(self,folder):
        NotImplementedError()


#---------------- TBD ----------------

def saveJSON(filename, data):
    h,_ = path.split(filename)
    if not path.exists(h):
        mkdir(h)
    with open(filename, "w") as jsonfile:
        json.dump(data,jsonfile, indent=4)

def loadJson(filename):
    result = dict()
    with open(filename, "r") as jsonfile:
         result  = json.load(jsonfile)
    
    return result

def tensorToNpArray(tens:Tensor)->np.ndarray:
    arr = tens.numpy()
    # Dimensionen des numpy-arrays müssen rotiert werden

def saveListOfListsToGrid(nd_list:list,filename,rows):
    tensorlist = [Tensor(npArraytoTensor(array)) for it in nd_list for array in it]
    grid = utils.make_grid(tensor=tensorlist,nrow=rows)
    utils.save_image(tensor=grid,fp=path.join(filename))

def saveListOfListsToCSV(list, filename):

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(list)

def readOutput(filename)->list:
    result = []
    with open(filename, 'r') as f:
        reader = csv.reader(f,delimiter=",")
        for row in reader:
            result.append(row)

    return result

def npArraytoTensor(arr:np.ndarray)->Tensor:
    """
        Erstellung eines Torchtensors, der Bildwerte repräsentiert
        Args:
        arr: np-Array mit Bildwerten in shape (height, width ,channels)
        Return:
        Torch-Tensor mit Bildwerten in shape (channels, height, width)
    """
    #arrnp.transpose(arr, (0, 3, 1, 2)).astype(np.float32)# Dimensionen für numpy-arrays müssen rotiert werden
    arr2 = np.transpose(arr, (2, 0, 1)).astype(np.float32)
    return Tensor(arr2)

def repl(string):
    result = string.replace(" ","_")
    result = result.replace("/","_")
    result = result.replace("ä","ae")
    result = result.replace("ü","ue")
    result = result.replace("ö","oe")
    result = result.replace("ß","ss")

    return result