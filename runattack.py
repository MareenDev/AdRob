from argparse import ArgumentParser
from classes.framework import DataHandler, AttackProtocol
from classes.models import ImageDecision
from classes.attack import HSJ, SignOPT
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
    parser.add_argument('--refname', type=str, default='Test', help= 'Name of entry in protocol')
    # attack 
    parser.add_argument('--attack_name', type=str, default='HSJ', help= 'Attack to be used for generating adversarial examples', required=True)
    parser.add_argument('--metric', type=str, help='metric to use in attack, e.g. linf or l2 ',default="")
    parser.add_argument( '--params_bool', type=lambda x: {str(k):bool(int(v)) for k,v in (i.split(':') for i in x.split(','))},
        help='comma-separated field:position pairs, e.g. targeted:True,...'
    )
    parser.add_argument( '--params_int', type=lambda x: {str(k):v.replace("[","").replace("]","") for k,v in (i.split(':') for i in x.split(','))},
        help='comma-separated field:position pairs, e.g. targeted:True,...', default={}
    )
    parser.add_argument( '--params_float', type=lambda x: {str(k):v.replace("[","").replace("]","") for k,v in (i.split(':') for i in x.split(','))},
        help='comma-separated field:position pairs, e.g. targeted:True,...', default={}
    )

    args = parser.parse_args()
    for key in args.params_int:
        values = args.params_int[key].split(";")
        args.params_int[key] = [int(v) for v in values]

    for key in args.params_float:
        values = args.params_float[key].split(";")
        args.params_float[key] = [float(v) for v in values]

    return args

def _checkInput(args):
    #Ausbaustufe: Prüfung, anhand von Mapping, ob 
    # alle parameter für Angriff gesetzt sind
    result = True

    return result

def main(args):
    attProcessingData = AttackProtocol(id=args.refname)

    # load data as np-array
    npArrayH = DataHandler()
    npArrayH.load(directory=args.data_dir)
    
    data = npArrayH.getData()
    shape = npArrayH.getShape()

    attProcessingData.setDataset(name=npArrayH.getName(),size = npArrayH.getSize(), shape=shape)
    # load API-Call Handler
    if args.requestHandlerClass =="ImageDecision":
        apiCall = ImageDecision(url=args.requestURL, method=args.requestMethod,id=args.refname, labels=args.labels)
        attProcessingData.setApiData(handlerClass=args.requestHandlerClass,method=args.requestMethod,url=args.requestURL)
    else:
        raise NotImplementedError()
    
    # load attack
    batchsize = 1 # Zunächst nur Verarbeitung mit Batchsize 1 möglich
    paramCombination = []

    if args.attack_name == "HSJ":

    # Prepare parameter-combination
        for i in args.params_int['iter']:
            for j in args.params_int['iter_max']:
                for k in  args.params_int['eval_init']:
                    for l in  args.params_int['eval_max']:
                        paramCombination.append({'iter':i, 'iter_max':j,'eval_init':k,'eval_max':l})
        print(paramCombination)
        if args.metric == "linf":
            metric = np.inf   
        else: 
            metric = 2

        attack = HSJ(name=args.refname,apiCall= apiCall,shape=shape,batchsize=batchsize, norm=metric,
                     targeted= args.params_bool['targeted'], verbose=args.params_bool['verbose'] )
        """attack = HSJ(name=args.refname,apiCall= apiCall,shape=shape,targeted= args.params_bool['targeted'],batchsize=batchsize, norm=norm, 
                            max_iter=0, max_eval=args.params_int['eval_max'], init_eval= args.params_int['eval_init'],
                            verbose=args.params_bool['verbose'] )"""
        
    elif args.attack_name == "SignOpt":

    # Prepare parameter-combination
        for i in args.params_int['iter_max']:
            for j in args.params_int['num_trial']:
                for k in  args.params_int['k']:
                    for l in  args.params_int['query_limit']:
                        for m in  args.params_float['alpha']:
                            for n in  args.params_float['beta']:
                                for o in args.params_float['epsilon']:
                                    paramCombination.append({'iter_max':i, 'num_trial':j,'k':k,'query_limit':l,'alpha':m,'beta':n,'eval_perform':args.params_bool['eval_perform'],'epsilon':o})
        
        attack = SignOPT(name=args.refname,apiCall=apiCall,shape=shape,norm=args.metric,targeted=args.params_bool['targeted'],batchsize=batchsize,
                        verbose=args.params_bool['verbose'])

    else:
        raise NotImplementedError()
    
    attProcessingData.setAttackData(name=args.attack_name,targeted=args.params_bool['targeted'],norm=args.metric)

    c = 0
    # run attacks
    for params in tqdm(paramCombination):
        attackStart = time.time()
        for key,value in data.items(): 
            c+=1
            for i,x_ref in enumerate(value["input"]): 
                x_ref2 = np.expand_dims(x_ref, axis=0) # expand shape: (Height,Width,Channels) -> (batchsize=1,Height,Width,Channels) for BBClassifier
                name = basename(key).replace(".npy","_")+str(i)
                attack.generate(x_ref=x_ref2,name=name,**params)
                
                # save attack data and reset lists for next parameter combination
        duration = time.time() - attackStart
        dir = attack.saveData(folder=join("data","attacks"))
        attack.initLists()

        attProcessingData.addRTData(folder=dir,parameter=params,runtime=duration)
        if c%200 == 0:
            attProcessingData.save()

    attProcessingData.stop()
    attProcessingData.save()    
    print("Gesamtlaufzeit",time.time() - attProcessingData.getStart())

if __name__ == "__main__":
    args = _parseInput()
    if _checkInput(args=args) :
        main(args)
    else:
        print("TBD")