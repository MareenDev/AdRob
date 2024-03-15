from os import path, mkdir, listdir
import matplotlib.pyplot as plt
#import seaborn as sns

import numpy as np
import time
import json
import requests

class JSONHandler:
    def __init__(self) -> None:
        self.data = dict()

    def setData(self, data):
        self.data = data

    def getData(self):
        return self.data

    def load(self, filename):
        if path.isfile(filename):   
            with open(filename, "r") as jsonfile:
                self.data  = json.load(jsonfile)
        else:
            raise ValueError("Given folder doesn't contain a file")

    def save(self, filename):
        h,_ = path.split(filename)
        if not path.exists(h):
            mkdir(h)
        with open(filename, "w") as jsonfile:
            json.dump(self.data,jsonfile, indent=4)

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
        jH = JSONHandler()
        filename = path.join(directory,"dataset.json")
        jH.load(filename)
        dataset = jH.getData()

        self.name = dataset["name"]
        self.shape = dataset["shape"]
        self.size = dataset["size"]
        data = dict()
        labels = dict()
        for element in dataset["data"]: 
            data[element["path"]]= {"input":np.load(element["path"]), "label":element["label"]} #Wichtig: x_ref daten müssen an Position 0 steht!!!
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
            np.save(arr=self.data[key]["input"],file=filename)

class AttackProtocol:
    def __init__(self,id):    
        self.filename = path.join("data","attacks",id,"protocol_"+id+".json")
        self.start    = time.time()
        self.end      = None

        self.id         = id
        if path.exists(self.filename):
            self.load() # Werfe Fehlermeldung! und achte darauf, dass in der Angriffsausführung bei zwischenspeicherung die load methode explizit ausgeführt wird
        else:
            self.aName      = ""
            self.aTargeted   = bool()
            self.aNorm       = None
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

    def setAttackData(self,name:str,targeted:bool,norm:str):
        self.aName      = name
        self.aTargeted  = targeted
        self.aNorm      = norm
        
    def setApiData(self,handlerClass,url,method):
        self.reqHandlerClass = handlerClass
        self.reqestUrl  = url
        self.reqMethod  = method
        
    def setDataset(self,name,size,shape):
        self.dataset = { "name": name, "size":size, "shape":shape}

    def getRTData(self):
        return self.RTData

    def addRTData(self,folder,parameter, runtime):
        self.RTData[str(self.counter)] = {"folder":folder,
                                        "parameter":parameter,
                                        "runtime":self._getDurationFormatted(runtime)}
        self.counter +=1

    def stop(self):
        self.end = time.time()

    def getStart(self):
        return self.start
    
    def getDataset(self):
        return self.dataset

    def getAName(self):
        return self.aName
    
    def _getDurationFormatted(self,seconds):
        return str(int(seconds // 3600))+"h "+str(int((seconds % 3600)//60))+"min "+str(int(seconds % 60))+"sek"

    def save(self):
        #Check if Protocol already exist
        jh = JSONHandler()
        data = {"ID":self.id,
                "Attack":self.aName,
                "API":{
                    "Handler":self.reqHandlerClass,
                    "URL":self.reqestUrl,
                    "Method":self.reqMethod
                    },
                "Dataset":self.dataset,
                "Runs":self.RTData}      
        jh.setData(data)
        jh.save(self.filename)  
        #saveJSON(data=data,filename=self.filename)

    def load(self):
        jH = JSONHandler()
        jH.load(self.filename)
        data = jH.getData()
        self.aName = data["Attack"]
        self.counter    = len(data["Runs"])
        self.id         = data["ID"]
        self.reqHandlerClass = data["API"]["Handler"]
        self.reqestUrl  = data["API"]["URL"]
        self.reqMethod  = data["API"]["Method"]
        self.dataset    = data["Dataset"]
        self.RTData     = data["Runs"]

    def getAttackPaths(self):
        result = []
        for _,value in self.RTData.items():
            result.append(value["folder"]) 
        return result

    def getDataByPath(self):
        result = dict()

        output = dict()
        input =  dict()
        queryCall =  dict()

        paths = self.getAttackPaths()        
        for key in paths:
            aL = AttackDataLoader(key)
            output = aL.getOutput()
            input = aL.getInput()        
            queryCall = aL.getQueryCalls()
            labels = self._getLabels()
            result[key]=dict()

            if len(output) == len(input) == len(queryCall):
                for k in output:
                    result[key][k] = {"output":output[k],"input":input[k],"queryCalls":queryCall[k],"labels":labels[k.split("_")[0]]}
            else:
                raise ValueError()
        return result
    
    def _getLabels(self):
        result = dict()

        dataset = self.getDataset() 
        dL = DataHandler()
        dL.load(path.join(dataset["name"],"Dataset"))
        data = dL.getData()
        for k in data :
            result[path.basename(k)[:-4]] = data[k]["label"] #Neuer Key erhält nur dateinamen, ohne Pfad und Dateiendung
        return result
    
class MultiPlot:
    def __init__(self,x_label:str,y_label:str,size, x_step,y_step,x_max,y_max,title):
        self.colors =  ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf','#bccd22', '#17ff1f']
        self.colorMap = dict()
        self.fig = self.init_plot(size=size,x_label=x_label,y_label=y_label,x_max=x_max,y_max=y_max,x_step=x_step,y_step=y_step,title=title)

    def init_plot(self,size,x_label,y_label,x_max,y_max,x_step,y_step,title):
        plt.ioff()
        
        fig = plt.figure(figsize=size)
        plt.yticks(np.arange(0, y_max, step=y_step)) # Values in [0,y_max] 
        plt.xticks(np.arange(0, x_max+x_step,step=x_step))#self.x_sorted[-1]+steps, step=steps))
        plt.xlabel(xlabel=x_label)
        plt.ylabel(ylabel=y_label)
        plt.title(title)

        return fig
    
    def _mapColor(self,name):
        result = ""
        try:
            result = self.colorMap[name]
        except:
            i = len(self.colorMap)
            try:
                result = self.colors[i]
                self.colorMap[name]=result
            except:
                raise ValueError("Allready 10 datasources used. Adding more is not allowed.")
        return result
    
    def addData(self,name:str,x:list,y:list, useLowerBound = False, usePoinLabels =False,pointlabels=[]):
        if len(x) != len(y):
            raise ValueError("Lists x and y need to have same length")

        #plot points with given color
        color = self._mapColor(name)
        plt.scatter(x,y, c=color, alpha=0.5,label=name)
        plt.legend(loc="upper left")

        if useLowerBound:
            self._addLowerBoundCurves(name,x,y)
        if usePoinLabels:
            self._addPointLabels(pointlabels,x,y)
    
    def _getPointWithMinY(self,points:list):
        points.sort(key=lambda x: x[1])  
        result = points[0]
        return result

    def _addLowerBoundCurves(self,name:str,x:list,y:list):
        color =self._mapColor(name)
        points = []

        x_curve = []
        y_curve = []

        #Create points and sort them among x (increasing)
        for i,_ in enumerate(x):
            points.append((x[i],y[i]))

        points.sort(key=lambda x: x[0])  


        #if serveral point with same x-value occure, just collect those with lowest y-value
        binaerStr = "0" 
        for i in range(1,len(points)):
            if points[(i-1)][0]==points[i][0]:
                binaerStr+="1"
            else:
                binaerStr+="0"
        
        if binaerStr.find("01")<0: #everythings fine
            x_curve = [x for x,_ in points]
            y_curve = [y for _,y in points]
        else: #filter multiple points for x
            p = [] 
            idx_s = [i for i in range(len(binaerStr)) if binaerStr.startswith('01', i)]
            idx_e = [i+1 for i in range(len(binaerStr)) if binaerStr.startswith('10', i)]

            if len(idx_e) < len(idx_s):
                idx_e.append(len(points)-1)

            for i,ch in enumerate(binaerStr):
                if (ch == '0') and (i not in idx_s):
                    p.append(points[i])

            for i,idx in enumerate(idx_s):
                 p.append(self._getPointWithMinY(points[idx:idx_e[i]]))

            p.sort(key=lambda x: x[0])  
            x_curve = [x for x,_ in p]
            y_curve = [y for _,y in p]

        plt.plot(x_curve,y_curve,color=color)

    def _addPointLabels(self,names:list,x:list,y:list):
        if len(names) == len(x):
            for i,_ in enumerate(x):
                plt.text(x[i],y[i],path.basename(names[i]))

    def show(self):
        self.fig.show()

    def save(self, filename ):
        #filepath = path.dirname(filename)
        #p = PathHandler(filepath)
        #p.create_subdirectory(path.basename(path.dirname(filepath)))
        self.fig.savefig(filename)
#---------------- classes to inherent from for extention----------------

class Collector:
    """ Abstract class (as interface) for collecting data
        inherent class for different kind of data 
        or different data source 
    """
        
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
        dataset = {"name":directory, "size":self.size,"shape":self.shape, "data":[]}
        jH = JSONHandler()
        for key in data:
            filename = path.join(subdir,path.basename(key)+".npy")    
            np.save(file=filename,arr=data[key])
            dataset["data"].append({"path":filename, "label": path.basename(key)})# Probleme mit referenzobjekt.

        jH.setData(dataset)
        jH.save(path.join(subdir,"dataset.json"))
            
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
                d[key]={"input": value,"label":"NotAvailable"}
#                dh.setData(data={key:value}) #Wichtig: x_ref daten müssen an 'Position 0' steht!!!      
            dh.setData(d)
            dh.saveData(inputfolder)  

            #Save model-output
            y_dict = dict()
            o_dict = dict()
            for key in self.y_list:
                y_dict[key] = [str(x) for x in self.y_list[key]] 
                o_dict[key] = [str(x) for x in self.querycount[key]]
            jH = JSONHandler()
            jH.setData(y_dict)
            jH.save(path.join(attackfolder,"Output.json"))

            jH.setData(o_dict)
            jH.save(path.join(attackfolder,"APICalls.json"))

#            saveJSON(data=y_dict,filename=path.join(attackfolder,"Output.json"))
#            saveJSON(data=o_dict,filename=path.join(attackfolder,"APICalls.json"))

        else: 
            attackfolder = None
        return attackfolder

class AttackDataLoader:
    def __init__(self,folder) -> None:
        self.folder = folder

        self.output = self._loadOutput()
        self.queryCalls= self._loadQueryCalls() 
        self.input = self._loadInput()

    def getRefname(self):
        return path.basename(self.folder)

    def getOutput(self):
        return self.output
    
    def getInput(self):
        return self.input

    def getQueryCalls(self):
        return self.queryCalls
    
    def _loadOutput(self):
        try:
            result = dict()
            jH = JSONHandler()
            jH.load(path.join(self.folder,"Output.json"))

            data = jH.getData()
            for k,v in data.items():#outputs
                result[k]={"ref": v[0],"advList":v[1:]}

            return result
        except:
            raise FileNotFoundError("No output-file (Output.json) available.")

    def _loadQueryCalls(self):
        try:
            result = dict()
            jH = JSONHandler()
            jH.load(path.join(self.folder,"APICalls.json"))

            data = jH.getData()
            for k,v in data.items():#outputs
                result[k]={"ref": v[0],"advList":v[1:]}
            return result
        except:
            raise FileNotFoundError("No output-file (APICall.json) available.")
        
    def _loadInput(self):
        try:
            result = dict()
            ph = PathHandler(path.join(self.folder,"Input"))
            filenames = ph.get_filenames("npy")
            for f,liste in filenames.items():
                for filename in liste:
                    liste = np.load(path.join(f,filename))
                    #result[path.join(f,filename)] = {"xRef":liste[0],"xAdvList":liste[1:]}#np.load(path.join(f,filename))
                    result[path.basename(filename)[:-4]] = {"ref":liste[0],"advList":liste[1:]}#np.load(path.join(f,filename))
            
            return result
        except:
            raise NotADirectoryError("No directory", path.join(self.folder,"Input"),"available.")
        
    
class Interpreter:
    def __init__(self,refname,**kwargs):
        self.refname = refname
        self.name = ""
        self.description = ""

    def get(self):
        raise NotImplementedError()
    

    def save(self,folder):
        raise NotImplementedError()
 
    #---------------- TBD ----------------

def repl(string):
    result = string.replace(" ","_")
    result = result.replace("/","_")
    result = result.replace("ä","ae")
    result = result.replace("ü","ue")
    result = result.replace("ö","oe")
    result = result.replace("ß","ss")

    return result