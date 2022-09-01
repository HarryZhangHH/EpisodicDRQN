import itertools
import torch
import numpy as np

def decode_one_hot(state):
    decode = 0
    for i in range(state.shape[0]):
        decode += state[i]*2**i
    if type(decode) == int:
        decode = torch.tensor(decode)
    return decode.long()
        
def argmax(x):
    denominator = 1000000
    if torch.is_tensor(x):
        return torch.argmax(x + torch.rand(x.shape[-1])/denominator)
    elif type(x) is np.ndarray:
        return np.argmax(x + np.random.rand(x.shape[-1])/denominator)
    elif type(x) == list:
        x = np.asarray(x)
        return np.argmax(x + np.random.rand(x.shape[-1])/denominator)

def iterate_combination(n):
    idx = [i for i in range(n)]
    iter = []
    for i in range(1, n+1):
        iter.extend(list(itertools.combinations(idx,i)))
    return iter

