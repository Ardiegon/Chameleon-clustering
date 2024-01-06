import os
import argparse

from src.generate_data import RawData, RawDataConfig
from src.chameleon import chameleon
from src.visualizers import visualise_hyperplane

def parse_args():
    pass

def delete_png_files(folder_path):
    files = os.listdir(folder_path)

    for file in files:
        if file.endswith('.png'):
            file_path = os.path.join(folder_path, file)
            os.remove(file_path)


def main(args):
    # rd = RawData(RawDataConfig(from_file="data/data_01.pickle"))
    rd = RawData(RawDataConfig(4,20,3))
    delete_png_files("plots")
    visualise_hyperplane(rd, [0,1], "plots/intro.png")
    answers, n_clusters = chameleon(rd, 4, 10, 4, 1.0, plot=True)
    print(n_clusters)
    visualise_hyperplane(rd, [0,1], "plots/outro.png", other_labels=answers)


if __name__ == "__main__":
    args = parse_args()
    main(args)