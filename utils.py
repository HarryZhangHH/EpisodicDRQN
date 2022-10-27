import itertools
import torch
import numpy as np

def question(q):
    i = 0
    while i < 2:
        answer = input(f"{q}? [y/n]")
        if any(answer.lower() == f for f in ["yes", 'y', '0', 'ye']):
            return True
            break
        elif any(answer.lower() == f for f in ['no', 'n', '1']):
            return False
            break
        else:
            i += 1
            if i < 2:
                print('Please enter yes or no')
            else:
                print("Default setting - no")
                return False

def label_encode(state):
    decode = 0
    if type(state) == list:
        for i in range(len(state)):
            decode += state[i]*2**i
    elif type(state) is np.ndarray or torch.is_tensor(state):
        for i in range(state.shape[0]):
            decode += state[i]*2**i
    if type(decode) == int:
        decode = torch.tensor(decode)
    return decode.long()
        
def argmax(x):
    denominator = 1000000
    if torch.is_tensor(x):
        with torch.no_grad():
            return torch.argmax(x + torch.rand(x.shape[-1])/denominator).item()
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

