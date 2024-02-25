import bs4
from os import path
from PIL import Image
import numpy as np
from classes.framework import Collector, DataPreparation, PathHandler, RestApiCall,saveJSON
import requests


#Classes for Collecting Data
    
class ImageByGoogle(Collector):
    def __init__(self) -> None:
        super().__init__()
        self.labels = []
        self.number = 0

    def setLabels(self,labels:list=[]):
        self.labels = labels
    
    def setNumber(self,number):
        self.number = number

    def collectData(self):
        """ Collect images by google-search for each label in labels
        Args:
            number: Number of images to be saved for each label.
            labels: list of labels for image search 
        Return:
            Dictonary of Images in byteformat
        """
        
        result = dict()
        
        for searchTerm in self.labels:
            #1. Führe Suche aus
            #1.1 HTML-Code der Bildseite herunterladen
            url = "https://www.google.de/search?tbm=isch&q="+searchTerm
            response = requests.get(url)
            html = response.content
            #1.2 BeautifulSoup-Objekt für die HTML-Seite erstellen und Bilder auslesen
            soup = bs4.BeautifulSoup(html, "html.parser")
            image_tags = soup.find_all("img") 

            #Hinweis: Mit der Suche können nur 21 Bilder gefunden werden
            if len(image_tags)<self.number-1 : 
                num = len(image_tags)
                print("Es wurden nur ",str(len(image_tags))," Bilder zum Suchterm ", searchTerm, "gefunden.") 
            else:
                num = self.number

            result[searchTerm] = []

            #2. Erstelle Liste aller Bilder 
            for i in range(1,num+1): #Erstes Bild des Suchergebnisses kann nicht verarbeitet werden. -> Shift um 1
                try:
                    image_tag = image_tags[i]
                    image_url = image_tag["src"]
                    response = requests.get(image_url)
                    image = response.content

                    result[searchTerm] += [image]
                except:
                    pass
        return result

    def saveData(self, data:dict,folder)->None:
        """ Save Images 
        Args:
            data: Dictonary of imagedata as bytedata. Key values: folder, values:imagedata 
        """
        for key in data:
            #0. Erstelle einen Unterordner je Suchbegriff
            ph = PathHandler(folder)
            sub_folder = ph.create_subdirectory(key)

            #Hinweis: Mit der Suche können nur 21 Bilder gefunden werden
    
            #2. Speichere Bilder in directory ab
            for i,image in enumerate(data[key]): 
                filename = path.join(sub_folder,str(i)+".jpg")
                with open(filename, "wb") as f:
                    f.write(image)
 
class ImageByTVDataset(Collector):

    def __init__(self) -> None:
        super().__init__()
        self.number = 0
        self.folder = None

    def setNumber(self,number):
        self.number = number

    def setDataset(self,dataset):
        self.dataset = dataset

    def collectData(self)->dict:
        result = dict()

        #TBD Reduziere dataset auf anzahl number
        for i in self.dataset.classes:
            result[i]=[]

        for x,y in self.dataset:
            i = self.dataset.classes[y]
            result[i]+=[x]

        for key in result:
            result[key] = result[key][:self.number]

        return result
    
    def saveData(self, data:dict,folder)->None:
        ph =PathHandler(folder)
        for key in data:
            #0. Erstelle einen Unterordner je Suchbegriff
            sub_folder = ph.create_subdirectory(key)
   
            #1. Speichere Bilder in directory ab
            for i,image in enumerate(data[key]): 
                filename = path.join(sub_folder,str(i)+".jpg")
                image.save(filename)

class ImageByModelQuery(Collector):
    def __init__(self) -> None:
        super().__init__()
        self.apiCall = None

    def collectData(self,inputdata:dict)->dict:          
        result = dict()
        for key,value in inputdata.items():
            for i,x in enumerate(value):
                y = self.apiCall.predict(x=x)
                result[key+str(i+1)]=y
        return result

    def setRequestHandler(self,apiCall:RestApiCall):
        self.apiCall = apiCall


#Classes for Preparing datacollection

class ImagePrep(DataPreparation):

    def prepare(self, shape):
        result = dict()
        size = (shape[0],shape[1])

        #reshaping images to (shape[0],shape[1],3)
        for folder in self.data:
            for filename, img in self.data[folder].items():
                if len(img.getbands())!=3:
                    img = img.convert('RGB')
                img = img.resize(size, Image.ANTIALIAS)
                img.save(path.join(folder,filename)) 
        self.shape = (shape[0],shape[1],3)

        for folder,namedImages in self.data.items():#erste key-Ebene folder, zweite data
            result[path.basename(folder)]=[np.array(namedImages[key], dtype=np.float32)/255 for key in namedImages]
        
        return result

    def load(self,directory):
        """ 
            get a all images in self.ph.path
            Return:  
                directory with imagedata, key: folder, value:Dictionary: key filename value PIL-Images)
        """
        # Search all images in folder 
        ph = PathHandler(directory)
        imagesJPG = ph.get_filenames("jpg")
        imagesPNG = ph.get_filenames("png")
        
        #Create dictonary of image-data
        imageNameDic = self._getUnionOfDictionarys(imagesJPG, imagesPNG)

        for key in imageNameDic:
            self.data[key] = dict()
            for imageName in imageNameDic[key]:
                imgageData = Image.open(path.join(key,imageName))#daten müssen 3channel haben!

                #result[path.join(key,imageName)] =imgageData 
                self.data[key][imageName] = imgageData 
                self.size += 1
        
    def _getUnionOfDictionarys(self,dict1:dict,dict2:dict):
        keys = set()
        result = dict()

        keys.update(dict1.keys())
        keys.update(dict2.keys())

        for key in keys:
            try: 
                result[key] = dict1[key]
            except: 
                pass
            try: 
                result[key] += dict2[key]
            except: 
                pass

        return result
           
    def saveImageByNPArray(self,name,array):
        img = self.getImageByNPArray(array)
        img.save(name)

    def getImageByNPArray(self,array):
        #Sollen NP-Arrays mittels PIL gespeichert werden ist eine Umskalierung notwendig: von float32->int8 
        array_new = array*255
        return Image.fromarray(obj=array_new.squeeze().astype(np.uint8),mode='RGB')

def loadAttack(protocolname):
    pass