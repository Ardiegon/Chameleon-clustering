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
    parser.add_argument("--plot", "-p", default=0, type=int, choices=[0,1,2], help="which things should be plotted")
    parser.add_argument("--target-clusters-amount", "-t", type=int, required=True) 
    parser.add_argument("--alpha", "-a", type=float, default=2.5) 
    parser.add_argument("--min-size", "-m", default = 10, type=int,
                        help="Minimum wanted amount of data points in cluster.")
    parser.add_argument("--k-nearest-neighbors", "-k", default=10, type=int)
    parser.add_argument("--automatic-min-size", "-A", action="store_true")
    parser.add_argument("--calculate-metrics", "-M", action="store_true")
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
    
    # rd = RawData(RawDataConfig(from_file="data/data_01.pickle"))
    rd = RawData(RawDataConfig(from_file="data/seeds.txt"))
    # rd = RawData(RawDataConfig(from_uci=UcimlDatatypes.WINE))
    # rd = RawData(RawDataConfig(4,200,5))
    # rd = RawData(RawDataConfig(4,20,5))
    
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

    print("Sorting chameleon predictions accordingly to order of original data...")
    answers = sort_predictions(rd.data, answers)

    if len(np.unique(answers[1])) == len(np.unique(rd.labels)):
        answers[1] = align_prediction_keys(rd.labels, answers[1])

        if args.calculate_metrics:
            print(f"Elapsed time: {elapsed:.3f}")
            _, metrics = compute_confusion_matrix_and_metrics(rd.labels, answers[1], "plots/cm.png")
            print(metrics)
    
    if args.plot:
        visualise_hyperplane(rd, [0,1], "plots/intro.png")
        visualise_hyperplane(answers, [0,1], "plots/outro.png")


if __name__ == "__main__":
    args = parse_args()
    main(args)