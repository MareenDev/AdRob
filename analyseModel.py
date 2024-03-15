from interpret import RobustnessCurve, AccuracyPerturbationCurve,AdvExampleGrids,ImagePerturbation
from classes.framework import AttackProtocol
import numpy as np
from argparse import ArgumentParser
from os.path import join, basename

def _parseInput():
    parser = ArgumentParser()
    parser.add_argument('--refname', help='attack-data (subfolders in path ./data/attacks/)', type=str) 
    parser.add_argument("--domain", type=str, default="Image", help="Domain for data")
    parser.add_argument('--figuresize', default=(50,10), help='Values for width and height of the plot', type=lambda s: tuple([int(item) for item in s.split(',')])) 
    parser.add_argument('--x_max', default=30, help='max value on x-axis', type=int) 
    parser.add_argument('--y_max', default=1, help='max value on y-axis', type=int) 
    parser.add_argument('--x_step', default=0.5, help='steps in x-axis', type=float)
    parser.add_argument('--y_step', default=0.1, help='steps in y-axis', type=float) 
    parser.add_argument('--interpreter', help='',default="GetCraftedSamples", nargs='?', choices="[RobustnessCurve,GetCraftedSamples,AccuracyPerturbationCurve,ImagePerturbation]")#["CleanAccuracy","RobustAccuracyL2","RobustAccuracyLinf","RobustAccuracyQuery","GetCraftedSamples" ]) 
    parser.add_argument("--metric", type=str, default="L2", help="Distance-Metrik", choices="[L2,Linf]")
    parser.add_argument('--addLowerCurve', type= bool, default=False)
    parser.add_argument('--addConfigTexts', type= bool, default=False)
    args = parser.parse_args()

    return args

def main(args):
    folder = join("data","attacks",args.refname)
    protocolID = basename(folder)
    aP = AttackProtocol(protocolID)
    aP.load()
    data = aP.getDataByPath()

    if args.domain == "Image":
        if args.metric == "Linf":
            metric = np.inf
        else:
            metric = 2
        
        if "RobustnessCurve" in args.interpreter:
            interpreter = RobustnessCurve(args.refname, data = data, norm = metric)
            interpreter.createPlot(size=args.figuresize,x_step=args.x_step,y_step=args.y_step,x_max=args.x_max,y_max=args.y_max,
                                    useLowerBound=args.addLowerCurve,usePoinLabels=args.addConfigTexts)
        elif "AccuracyPerturbationCurve" in args.interpreter:
            interpreter = AccuracyPerturbationCurve(args.refname, data=data,norm = metric)
            interpreter.createPlot(size=args.figuresize,x_step=args.x_step,y_step=args.y_step,x_max=args.x_max,y_max=args.y_max,
                                    useLowerBound=args.addLowerCurve,usePoinLabels=args.addConfigTexts)
        
        elif "AccuracyQueryCount" in args.interpreter:
            #TBD
            #interpreter = AccuracyByCalls()
            #interpreter.loadData(folder)
            #interpreter.createPlot(size=args.figuresize,x_step=args.x_step,y_step=args.y_step,x_max=args.x_max,y_max=args.y_max,
            #                        useLowerBound=args.addLowerCurve,usePoinLabels=args.addConfigTexts)
            pass
        elif "GetCraftedSamples" in args.interpreter:
            #TBD Filtern der inputdaten für ref und 
            inputData = dict() 

            for ref, attackData in data.items():
                inputData[ref] = []
                
                for _,imagedata in attackData.items(): #Loop über die Inputelement-Referenzen
                    l1 = np.expand_dims(np.array(imagedata["input"]["ref"]), axis=0) 
                    l2 = imagedata["input"]["advList"]
                    d = np.vstack([l1,l2])                   
                    #TBD: Hier liegt ein Fehler vor! d enthält nur die Daten der advList!!!
                    inputData[ref].append(d)  
                
            interpreter = AdvExampleGrids(args.refname,data=inputData)

        elif "ImagePerturbation" in args.interpreter:
            interpreter = ImagePerturbation(args.refname, data = data)
        else:
            raise NotImplementedError()
        
        """
        elif "CleanAccuracy" in args.interpreter:
            ap = AttackProtocol(args.refname)

            #read Dataset-ref by protocolref
            path_d = join(ap.dataset["name"],"Dataset")
            dH = DataHandler()
            dH.load(path_d)
            input = dH.getData()

            #getModel
            model_clname = ap.reqHandlerClass
            model_method = ap.reqMethod
            model_url = ap.reqestUrl

            model_cls = vars(classes.models)["model_cl"]
            access = model_cls(url=model_url, method=model_method,id=args.refname, labels=args.labels)


            #calc output
            pass"""
        
        interpreter.save(join("data","interpretation"))
        print("Data saved at folder",join("data","interpretation",args.refname))

if __name__ == "__main__":
    args = _parseInput()
    main(args)
