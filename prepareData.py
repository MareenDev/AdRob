import argparse
from classes.data import ImagePrep


def _parseInput():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="Image", help="Domain for data")
    parser.add_argument("--shape",  help='delimited list input', type=lambda s: tuple([int(item) for item in s.split(',')])) 
    parser.add_argument('--data_dir', type=str, default='/data/dataset/Test', help= 'Directory for data to prepare')
    return parser.parse_args()

def main(args):
    if args.domain == "Image":
        dataH = ImagePrep()
    else: 
        raise ValueError("Domäne ", args.Domain," wird aktuell nicht unterstützt.") #Aktuell nur 'Image-Domäne' betrachtet
    
    dataH.load(directory=args.data_dir)
    dataToSave = dataH.prepare(shape=args.shape)
    dataH.createDataset(directory=args.data_dir,data=dataToSave)

if __name__ == "__main__":
    args = _parseInput()
    main(args)