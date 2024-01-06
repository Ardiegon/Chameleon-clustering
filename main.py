import os
import argparse

from src.generate_data import RawData, RawDataConfig
from src.chameleon import chameleon
from src.visualizers import visualise_hyperplane

DEF_CLUSTER_PARTITIONS = 10

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", "-p", action="store_true")
    parser.add_argument("--target-clusters-amount", "-t", type=int, required=True) 
    parser.add_argument("--min-size", "-m", default = 10, type=int,
                        help="Minimum wanted amount of data points in cluster.")
    parser.add_argument("--k_nearest_neighbors", "-k", default=10, type=int)
    parser.add_argument("--automatic-min-size", "-a", action="store_true")

    return parser.parse_args()

def delete_png_files(folder_path):
    files = os.listdir(folder_path)

    for file in files:
        if file.endswith('.png'):
            file_path = os.path.join(folder_path, file)
            os.remove(file_path)

def main(args):
    if args.plot:
        delete_png_files("plots")
    # rd = RawData(RawDataConfig(from_file="data/data_01.pickle"))
    rd = RawData(RawDataConfig(4,400,3))
    # rd = RawData(RawDataConfig(2,100,3))
    
    min_size = args.min_size
    target = args.target_clusters_amount
    n_neighbors = args.k_nearest_neighbors
    if args.automatic_min_size:
        n_data = len(rd.data)
        n_clusters = len(set(rd.labels))
        min_size = n_data/n_clusters/DEF_CLUSTER_PARTITIONS

    answers = chameleon(rd,target,n_neighbors,min_size,plot=args.plot)
    
    if args.plot:
        visualise_hyperplane(rd, [0,1], "plots/intro.png")
        visualise_hyperplane(answers, [0,1], "plots/outro.png")


if __name__ == "__main__":
    args = parse_args()
    main(args)