import numpy as np


def inteconectivity(cluster):
    pass

def average_weights(cluster):
    pass

def relative_interconectivity(c1, c2):
    """
    According to paper calculate how much 
    two clusters are alike in terms of shape
    """
    pass

def relative_closeness(c1,c2):
    """
    According to paper calculate how much
    two clusters are alike in terms of being close enough
    """

def cost_func(c1, c2, alpha):
    return relative_closeness(c1, c2)**alpha * relative_interconectivity
