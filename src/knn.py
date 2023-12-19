from sklearn.neighbors import NearestNeighbors
import numpy as np

from generate_data import load_data

X, y = load_data("example.pickle")
nbrs = NearestNeighbors(n_neighbors=5, algorithm='auto')
