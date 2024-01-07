import os
import argparse
import time
import numpy as np

from src.generate_data import RawData, RawDataConfig, UcimlDatatypes
from src.chameleon import chameleon
from src.visualizers import visualise_hyperplane
from src.utils import align_prediction_keys, sort_predictions, compute_confusion_matrix_and_metrics

DEF_CLUSTER_PARTITIONS = 10

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dr", nargs=3, default = [4,50,5], type=int, 
                        help ="Data will be generated randomly. Meaning is [number of classes, samples per class, sample dimensions]")
    parser.add_argument("--du", type=str, choices=[e.name for e in UcimlDatatypes], default = None, 
                        help ="Data will be downloaded from UCIML site. Choice is for now WINE or IRIS")
    parser.add_argument("--df", type=str, default = None,  
                        help ="Data will be loaded from file. It can be .pickle or .txt. If pickle, it should be prepared by RawData class, if txt, it should be tab separated file, where last column is labels.")
    parser.add_argument("--plot", "-p", default=0, type=int, choices=[0,1,2], 
                        help="which things should be plotted. 0 is nothing, 1 is only in and out state, 2 is creating GIF from each merge iteration (really slows up algorithm)")
    parser.add_argument("--target-clusters-amount", "-t", type=int, required=True,
                        help="how many clusters should be produced by chameleon.") 
    parser.add_argument("--alpha", "-a", type=float, default=2.5,
                        help="steers how much cost function should depend on interconectivity and closeness") 
    parser.add_argument("--min-size", "-m", default = 10, type=int,
                        help="Minimum wanted amount of data points in cluster.")
    parser.add_argument("--k-nearest-neighbors", "-k", default=10, type=int,
                        help="How many neighors will be searched while building graph")
    parser.add_argument("--automatic-min-size", "-A", action="store_true",
                        help="specify if you want progrm to choose automatically what should be minimum cluster size in first stage.")
    parser.add_argument("--calculate-metrics", "-M", action="store_true",
                        help="Writes to terminal elapsed time, metrics calculated from confusion matrix, and saves confusion matrix in plots directory")
    return parser.parse_args()

def delete_png_files(folder_path):
    files = os.listdir(folder_path)
    for file in files:
        if file.endswith('.png'):
            file_path = os.path.join(folder_path, file)
            os.remove(file_path)

def main(args):
    os.makedirs("plots", exist_ok=True)
    if args.plot:
        delete_png_files("plots")
    
    if args.du is not None:
        rd = RawData(RawDataConfig(from_uci=UcimlDatatypes[args.du]))
    elif args.df is not None:
        rd  = RawData(RawDataConfig(from_file=args.df))
    else:
        rd = RawData(RawDataConfig(args.dr[0], args.dr[1], args.dr[2]))
    
    min_size = args.min_size
    target = args.target_clusters_amount
    n_neighbors = args.k_nearest_neighbors
    if args.automatic_min_size:
        n_data = len(rd.data)
        min_size = n_data/target/DEF_CLUSTER_PARTITIONS

    should_be_parallel = args.plot<2
    
    start = time.time()
    answers = chameleon(rd, target,n_neighbors,min_size, alpha=args.alpha, plot=args.plot>=2, parallelism=should_be_parallel)
    elapsed = time.time()-start
    print(f"Elapsed time: {elapsed:.3f}")

    print("Sorting chameleon predictions accordingly to order of original data...")
    answers = sort_predictions(rd.data, answers)

    if len(np.unique(answers[1])) == len(np.unique(rd.labels)):
        print("Searching for best mapping betwen predicted labels names and originam label names...")
        answers[1] = align_prediction_keys(rd.labels, answers[1])

        if args.calculate_metrics:
            _, metrics = compute_confusion_matrix_and_metrics(rd.labels, answers[1], "plots/cm.png")
            print(metrics)
    
    if args.plot:
        visualise_hyperplane(rd, [0,1], "plots/intro.png")
        visualise_hyperplane(answers, [0,1], "plots/outro.png")


if __name__ == "__main__":
    args = parse_args()
    main(args)