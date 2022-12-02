import itertools
import torch
import numpy as np
import os, sys
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HiddenPrints:
    """
    To prevent a function from printing
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

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
    # if type(decode) == int:
    decode = torch.as_tensor(decode)
    return decode.long()
        
def argmax(x):
    denominator = 1000000
    if torch.is_tensor(x):
        x = x.to('cpu')
        with torch.no_grad():
            return torch.argmax(x + torch.rand(x.shape[-1])/denominator).item()
    elif type(x) is np.ndarray:
        return np.argmax(x + np.random.rand(x.shape[-1])/denominator)
    elif type(x) == list:
        x = np.asarray(x)
        return np.argmax(x + np.random.rand(x.shape[-1])/denominator)

def calculate_sum(x):
    if isinstance(x, int):
        return x
    elif torch.is_tensor(x):
        with torch.no_grad():
            return torch.sum(x).item()
    elif type(x) is np.ndarray:
        return np.sum(x)
    else:
        return sum(x)

def iterate_combination(n):
    idx = [i for i in range(n)]
    iter = []
    for i in range(1, n+1):
        iter.extend(list(itertools.combinations(idx,i)))
    return iter

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # env.seed(seed)
