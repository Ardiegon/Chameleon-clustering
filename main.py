import argparse

from src.generate_data import RawData, RawDataConfig
from src.chameleon import chameleon
from src.visualizers import visualise_hyperplane

def parse_args():
    pass

def main(args):
    rd = RawData(RawDataConfig(4,100,3))
    visualise_hyperplane(rd, [0,1], "zintro.png")
    answers, n_clusters = chameleon(rd, 4, plot=True)
    print(n_clusters)
    visualise_hyperplane(rd, [0,1], "zoutro.png", other_labels=answers)


if __name__ == "__main__":
    args = parse_args()
    main(args)