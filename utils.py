import os
import numpy as np
from icecream import ic

def ComputeQuantumEmbeddings(x):
    m = np.conj(x)
    return np.kron(x,m)