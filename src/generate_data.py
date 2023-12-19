import pickle
from typing import Any
import numpy as np

from dataclasses import dataclass
from utils import int_to_binary_cors
from visualizers import visualise_hypercube, visualise_hyperplane

def calc_default_vars(sample_dimensions, number_of_classes):
    return [[0.1 for j in range(sample_dimensions) ] for i in range(number_of_classes)]

def calc_random_vars(sample_dimensions, number_of_classes):
    return [np.random.rand(sample_dimensions) for i in range(number_of_classes)]

def calc_default_centers(sample_dimensions, number_of_classes):
    return [int_to_binary_cors(class_id, sample_dimensions) for class_id in range(number_of_classes)]

def calc_random_centers(sample_dimensions, number_of_classes, factor=5):
    return [np.random.rand(sample_dimensions)*factor for _ in range(number_of_classes)]


@dataclass
class RawDataConfig:
    def __init__(self, number_of_classes = None, number_samples = None, 
                 sample_dimensions = None, path = None, centers = None, 
                 vars = None, cluster_position_randomness = True) -> None:
        self.from_file = True if path is not None else False 

        if centers is None and path is None:
            centers = calc_default_centers(sample_dimensions, number_of_classes) if not cluster_position_randomness else calc_random_centers(sample_dimensions, number_of_classes)
        if vars is None and path is None:
            vars = calc_default_vars(sample_dimensions, number_of_classes) if not cluster_position_randomness else calc_random_vars(sample_dimensions, number_of_classes)
        
        self.info = {
            "path": path,
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
        with open(path, 'rb') as handle:
            self.data, self.labels = pickle.load(handle)

if __name__ == "__main__":
    # rdc = RawDataConfig(4, 100, 10)
    rdc = RawDataConfig(path = "example.pickle")
    rd = RawData(rdc)

    visualise_hyperplane(rd, (0,1), "hola01.png", color_classes=False)
    visualise_hyperplane(rd, (0,2), "hola02.png")
    visualise_hyperplane(rd, (1,2), "hola12.png")
    visualise_hypercube(rd, (0,1,2), "hola3D.png")
    rd.save_data("data.pickle")