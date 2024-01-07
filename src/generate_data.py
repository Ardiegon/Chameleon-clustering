import os
import pickle
import numpy as np
from typing import Any
from enum import Enum
import pandas as pd

import src.visualizers as vsl
from dataclasses import dataclass
from src.utils import int_to_binary_cors
from ucimlrepo import fetch_ucirepo 
  
  
def calc_default_vars(sample_dimensions, number_of_classes):
    return [[0.1 for j in range(sample_dimensions) ] for i in range(number_of_classes)]

def calc_random_vars(sample_dimensions, number_of_classes, factor):
    return [np.random.rand(sample_dimensions)*factor for i in range(number_of_classes)]

def calc_default_centers(sample_dimensions, number_of_classes):
    return [int_to_binary_cors(class_id, sample_dimensions) for class_id in range(number_of_classes)]

def calc_random_centers(sample_dimensions, number_of_classes, factor=5):
    return [np.random.rand(sample_dimensions)*factor for _ in range(number_of_classes)]

class UcimlDatatypes(Enum):
    IRIS=53
    WINE=109
    HEARTH=571

@dataclass
class RawDataConfig:
    def __init__(self, number_of_classes = None, number_samples = None, 
                 sample_dimensions = None, from_file = None, from_uci = None, centers = None, 
                 vars = None, cluster_position_randomness = True, closeness = 2.0, variation = 0.5) -> None:
        self.from_file = True if from_file is not None else False 
        self.from_uci = True if from_uci is not None else False 

        if centers is None and from_file is None and from_uci is None:
            centers = calc_default_centers(sample_dimensions, number_of_classes) if not cluster_position_randomness else calc_random_centers(sample_dimensions, number_of_classes, factor=closeness)
        if vars is None and from_file is None and from_uci is None:
            vars = calc_default_vars(sample_dimensions, number_of_classes) if not cluster_position_randomness else calc_random_vars(sample_dimensions, number_of_classes, factor=variation)
        
        self.info = {
            "path": from_file,
            "uci": from_uci,
            "centers": centers,
            "vars": vars,
            "number_of_classes": number_of_classes,
            "number_samples": number_samples,
            "sample_dimensions": sample_dimensions
        }

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """
        get info for data creation depending on given parameters
        """
        return self.info

class RawData:
    def __init__(self, config) -> None:
        if config.from_file:
            self.load_data(**config())
        elif config.from_uci:
            self.load_uci(**config())
        else:
            self.make_multidimensional_blobs(**config())
    
    def make_multidimensional_blobs(self, number_samples, sample_dimensions, number_of_classes, centers, vars, **kwargs):
        assert len(centers) == number_of_classes
        assert all(len(c) == sample_dimensions for c in centers)

        self.data = np.zeros((number_samples * number_of_classes, sample_dimensions))
        self.labels = np.zeros(number_samples * number_of_classes, dtype='int')

        for class_index in range(number_of_classes):
            start_index = class_index * number_samples
            end_index = (class_index + 1) * number_samples
            self.data[start_index:end_index, :] = np.random.normal(loc = centers[class_index], 
                                                        scale = vars[class_index], 
                                                        size = [number_samples, sample_dimensions])
            self.labels[start_index:end_index] = class_index

    def save_data(self, path):
        with open(path, 'wb') as handle:
            pickle.dump((self.data, self.labels), handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_data(self, path, **kwargs):
        _, file_extension = os.path.splitext(path)
        if file_extension==".pickle":
            with open(path, 'rb') as handle:
                self.data, self.labels = pickle.load(handle)
        elif file_extension==".txt":
            X = []
            y = [] 
            with open(path, 'r') as handle:
                for line in handle:
                    values = line.strip().split()
                    coordinates = list(map(float, values[:-1]))
                    class_label = int(values[-1])
                    X.append(coordinates)
                    y.append(class_label)
            mappings = {v:k for k,v in enumerate(np.unique(y))}
            self.labels = np.array([mappings[l] for l in y])
            self.data = np.array(X)
        else:
            raise(ValueError(f"Unknown data file type: {file_extension}"))

    def load_uci(self, uci, **kwargs):
        repo = fetch_ucirepo(id=uci.value) 
        data = repo.data.features
        # self.data = np.array([[float(v) for v in d] for d in data])
        self.data = self.clean_dataset(data)
        labels = repo.data.targets.to_numpy()  
        mappings = {v:k for k,v in enumerate(np.unique(labels))}
        self.labels = np.array([mappings[l[0]] for l in labels])

    def clean_dataset(self, df):
        string_columns = df.select_dtypes(include=['object']).columns
        df = pd.get_dummies(df, columns=string_columns)
        df = df.astype(float)
        data_array = df.values
        return data_array

if __name__ == "__main__":
    # rdc = RawDataConfig(3, 10, 4)
    # rdc = RawDataConfig(from_file = "data\\data_01.pickle")
    rdc = RawDataConfig(from_file = "data/smiley.txt")
    # rdc = RawDataConfig(from_uci = UcimlDatatypes.HEARTH)
    
    rd = RawData(rdc)
    
    print("Number of data dimensions: ", end="")
    print(len(rd.data[0]))
    print("Number of classes: ", end="")
    print(len(np.unique(rd.labels)))
    print("Names of classes: ", end="")
    print(np.unique(rd.labels))

    vsl.visualise_hyperplane(rd, (0,1), "hola01.png", color_classes=True)
    # vsl.visualise_hyperplane(rd, (0,2), "hola02.png")
    # vsl.visualise_hyperplane(rd, (1,2), "hola12.png")
    rd.save_data("data.pickle")