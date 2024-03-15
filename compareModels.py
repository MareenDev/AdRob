from interpret import Bundle,AccuracyPerturbationComp
from argparse import ArgumentParser
from classes.framework import AttackProtocol
from os import path
import numpy as np

def _parseInput():
    parser = ArgumentParser()
    parser.add_argument("--bundleName", type=str, help="")
    parser.add_argument('-l', '--refnames', help='comma-separated references for attack-data (subfolders in path ./data/attacks/)', type=lambda s: [str(item) for item in s.split(',')]) 
    parser.add_argument('--interpreter', help='',default="AccuracyPerturbationCurve", nargs='?', choices="[RobustnessCurve,GetCraftedSamples,AccuracyPerturbationCurve]")
    parser.add_argument("--metric", type=str, default="L2", help="Distance-Metrik", choices="[L2,Linf]")
    parser.add_argument("--domain", type=str, default="Image", help="Domain for data")
    parser.add_argument('--figuresize', default=(50,10), help='Values for width and height of the plot', type=lambda s: tuple([int(item) for item in s.split(',')])) 
    parser.add_argument('--x_max', default=30, help='max value on x-axis', type=int) 
    parser.add_argument('--y_max', default=1, help='max value on y-axis', type=int) 
    parser.add_argument('--x_step', default=0.5, help='steps in x-axis', type=float)
    parser.add_argument('--y_step', default=0.1, help='steps in y-axis', type=float) 

    args = parser.parse_args()

    return args

def main(args):
    bundle = Bundle(args.bundleName,args.refnames)
    paths = bundle.getPaths()
    data = dict()

    for p in paths:
        identifier = path.basename(p)
        aP = AttackProtocol(identifier)
        aP.load()
        data[identifier] = aP.getDataByPath()
        

    if args.domain == "Image":
        if args.metric == "Linf":
            metric = np.inf
        else:
            metric = 2
        
        #paths und interpreter
        if "AccuracyPerturbationCurve" in args.interpreter:
            interpreter = AccuracyPerturbationComp(refname=bundle.getName(),
                                                   data = data, norm = metric)
        elif "AccuracyQueryCount" in args.interpreter:
            #interpreter = Rob(refname=bundle.getName(),
            #                                       data = data, norm = metric)
            pass
        else:
            raise NotImplementedError()
        """elif "AccuracyQuery" in args.interpreter:
            interpreter = AccuracyByCallsVGL()"""
        
        #interpreter.loadData(bundle)
        interpreter.createPlot(size=args.figuresize,x_step=args.x_step,y_step=args.y_step,x_max=args.x_max,y_max=args.y_max)
        interpreter.save()
        print("Figure saved at folder",bundle.getFolder())

if __name__ == "__main__":
    args = _parseInput()
    main(args)
