# Miscellaneous functions
import numpy as np


def onehot_vectorize(j):
    temp=np.zeros((10,1))
    temp[j]=1.0
    return temp

def sigmoid_activation(x):
    return 1/(np.exp(-x)+1)

def sigmoid_prime(x):
    # S'=S*(1-S) derivative
    return sigmoid_activation(x)*(1-sigmoid_activation(x))