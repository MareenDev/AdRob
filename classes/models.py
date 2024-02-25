from torch import nn, Tensor, arange, float32
from classes.framework import RestApiCall,repl
import json
import numpy as np
import base64
from PIL import Image    

class ImageScore(RestApiCall):
    """ class for queriing api-classifiers resulting score-values"""

    def createRequestPayload(self, data):
        """Create json as request-payload containing image-data 
        Args: data - imagedata 
        Result: Json-string containing image-data"""
        try:  
            channels = len(data.getbands())
            image_size = data.size
            img_bytes = data.tobytes()
            img_s = base64.b64encode(img_bytes).decode()
            result = json.dumps({'image':img_s,'image_size':image_size, "channel":channels}) 
            return result
        except:
            raise ValueError("Creation of request payload failed. Please check inputdata",data)
        
    def createRequestHeader(self):
        """Create request-header   
        Result: request-header data"""
        try:
            result = {"ContentType":"application/json"}
            return result
        except:
            raise ValueError("Creation of request header failed. Please check implementation")
        
    def _getDataByResponse(self, resp):
        if self.responseIsOk(resp):
            try:
                data = json.loads(resp.content)['data']
                dataTensor = Tensor(data).unsqueeze(0) #batchsize = 1 wird hinzugefügt
                return dataTensor
            except:
                raise ValueError("Processing response fails. Please check response", resp)
        else:
            raise ValueError("Processing response fails. Please check response", resp)

    def _transformInput(self,x):
        """Transform input-data from numpyarray in PIL-image for further processing
        Args: numpy-array containing image data
        Return: PIL Image for x"""
        mode = 'RGB'
        data = np.squeeze(x, axis=0)                                      #delete batchsize = 1
        image = Image.fromarray(obj=(data*255).astype('uint8'),mode=mode) #ToCheck: Is there anpther way to con from np array to byte?
        return image 
     
class ImageDecision(ImageScore) :
    """ class for queriing api-classifiers resulting corresponding class"""

    def __init__(self , url , method='POST', id = "C", labels:list=[]) :
        super(ImageDecision, self).__init__(url, method, id)

        #label encoding
        numclasses = len(labels)
        onehotTensor = nn.functional.one_hot(arange(numclasses) , num_classes=numclasses)        
        labelEncoding = dict()
        for i,label in enumerate(labels):
            labelEncoding[repl(label)] = onehotTensor[i].to(float32)
        self.labelEncoding = labelEncoding

    def _getDataByResponse(self, resp):
        if self.responseIsOk(resp):
            body = json.loads(resp.content)
            data = body['data']
            dataTensor = Tensor()
            try:
                dataTensor = self.labelEncoding[repl(data)]
                dataTensor = dataTensor.unsqueeze(0)#batchsize = 1 wird hinzugefügt
                return dataTensor.numpy()
            except:
                raise ValueError("Cannot interpret label", repl(data),". Available Values:",list(self.labelEncoding.keys()))
        else:
            raise ValueError("TBD7")

    def getLabelEncoding(self):
        return self.labelEncoding
    