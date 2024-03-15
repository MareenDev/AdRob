import argparse
from classes.data import ImageByGoogle, ImageByTVDataset
from classes.framework import PathHandler
from torchvision.transforms import v2
from torchvision import datasets
from os.path import join, dirname, curdir

def _parseInput():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collector", type=str, default="ImageByGoogle", help="Domain for data")
    parser.add_argument("--dataset", type=str, default="FashionMNIST", help="Domain for data")
    parser.add_argument("--number", type=int, default="5", help="Number of data per label")
    parser.add_argument('--name', type=str, default='Test', help= 'Foldername for saving data')
    parser.add_argument('-l', '--labels', help='delimited list input', 
                        type=lambda s: [str(item) for item in s.split(',')]) 
    return parser.parse_args()

def main(args):


    ph = PathHandler(join(curdir,"data","input"))
    folder = ph.create_subdirectory(args.name)

    if args.collector == "ImageByGoogle":
        collector = ImageByGoogle()
        collector.setLabels(args.labels)
        collector.setNumber(args.number)
    elif args.collector == "ImageByTVDataset":
        transforms = v2.Compose([v2.ToPILImage()])
        if args.dataset =="FashionMNIST":
            dataset = datasets.FashionMNIST(root=folder,
                                            train=False, download=True, transform=transforms)
        elif args.dataset =="MNIST":
            dataset = datasets.MNIST(root=folder,
                                            train=False, download=True, transform=transforms)

        else:
            raise ValueError("Dataset", args.dataset," kann aktuell nicht verwendet werden.") 
        collector = ImageByTVDataset()
        collector.setDataset(dataset)
        collector.setNumber(args.number)
    else: 
        raise ValueError("Klasse", args.collector," kann aktuell nicht verwendet werden.") 
    
    data = collector.collectData()
    collector.saveData(data,folder)

if __name__ == "__main__":
    args = _parseInput()
    main(args)