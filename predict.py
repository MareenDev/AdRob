#load data
#quest schnittstelle
#save result as json
#load json ue 

from argparse import ArgumentParser
from classes.framework import DataHandler, AttackProtocol,JSONHandler,PathHandler
from classes.models import ImageDecision
from classes.attack import HSJ, SignOPT, GeoDAAttack
from os.path import join,basename
import time
import numpy as np
from tqdm import tqdm

def _parseInput():
    parser = ArgumentParser()

    # data settings
    parser.add_argument('--data_dir', type=str, help= 'Directory for prepared tensors', required=True)
    # RequestData
    parser.add_argument('-l', '--labels', help='delimited list input', type=lambda s: [str(item) for item in s.split(',')]) 
    parser.add_argument('--requestURL', type=str,  help= 'API-URL', required=True)
    parser.add_argument('--requestHandlerClass',type=str,  help= 'Class for RequestHandling', required=True)
    parser.add_argument('--requestMethod',type=str, default="POST", help= 'Class for RequestHandling')
    # protocol
    parser.add_argument('--baseID', type=str, default='Test', help= 'Name of entry in protocol')

    args = parser.parse_args()
    return args

def main(args):
    # load data as np-array
    modelID = args.requestURL

    pH = PathHandler("Test/Accuracy")
    pH.create_subdirectory(args.baseID)
    for j in range(10):
        npArrayH = DataHandler()
        npArrayH.load(directory=join(args.data_dir,str(j),"Dataset"))
        data = npArrayH.getData()

        # load API-Call Handler
        if args.requestHandlerClass =="ImageDecision":
            apiCall = ImageDecision(url=args.requestURL, method=args.requestMethod,id=args.baseID, labels=args.labels)
        else:
            raise NotImplementedError()
        
        d = dict()

        for _,value in data.items(): 
            label = value["label"]
            d[label]=[]
            for i,x_ref in enumerate(value["input"]): 
                x_ref2 = np.expand_dims(x_ref, axis=0) # expand shape: (Height,Width,Channels) -> (batchsize=1,Height,Width,Channels) for BBClassifier

                y_ref = apiCall.predict(x = x_ref2) 
                d[label].append(str(np.argmax(y_ref)))
                
        jH = JSONHandler()
        jH.setData(d)
        
        jH.save("Test/Accuracy/"+str(args.baseID)+"/"+str(args.baseID)+"_KL"+str(j)+".json")


if __name__ == "__main__":
    args = _parseInput()
    main(args)
    