import itertools
import torch
import numpy as np
import os, sys
import random
from typing import Union

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Type:
    # Represents a generic tensor type.
    # This could be an np.ndarray, tf.Tensor, or a torch.Tensor.
    TensorType = Union[np.array, "tf.Tensor", "torch.Tensor"]

    # Either a plain tensor, or a dict or tuple of tensors (or StructTensors).
    TensorStructType = Union[TensorType, dict, tuple]

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

def generate_features(agent: object, max_reward: float, max_play_times: float):
    """
    Generate extra features
    own_reward_ratio \in (0,1], own_defect_ratio \in (0,1], oppo_defect_ratio \in [0,1] , play_times_ratio \in [0,1]
    """
    own_reward = agent.running_score
    own_defect_ratio = calculate_sum(agent.own_memory)/agent.play_times
    oppo_defect_ratio = calculate_sum(agent.opponent_memory)/agent.play_times
    own_reward_ratio = own_reward/max_reward
    play_times_ratio = min(1, agent.play_times/max_play_times)
    return torch.FloatTensor([own_reward_ratio, own_defect_ratio, oppo_defect_ratio])

def generate_payoff_matrix(name:str = 'PD', REWARD:int = 1, TEMPTATION:int = None, SUCKER:int = None, PUNISHMENT:int = 0, N:int = 100, seed:int = 42):
    """
    Generate payoff matrix randomly

    Args:
        name: 'PD' or 'SH' or 'SD' or 'Ha'
              PD --> Prisoner's Dilemma     SH --> Stag Hunt    SD --> Snowdrift     Ha --> Harmony
                        S = 1
                  Harmony  |  Snowdrift
             0 ------------1-----------> T = 2
                 Stag Hunt | Prisoner's Dilemma
                         S = -1
    """
    assert name == 'PD' or name == 'SH' or name == 'SD' or name == 'Ha', f'Name {name} is wrong, please select one from ["PD","SH"]'
    if REWARD is not None and TEMPTATION is not None and SUCKER is not None and PUNISHMENT is not None:
        return REWARD, TEMPTATION, SUCKER, PUNISHMENT
    np.random.seed(seed)
    K = 0.2
    x = np.ones(N)
    reward = REWARD * x
    punishment = PUNISHMENT * x
    if name == 'PD':
        # prisoner's dilemma rule: TEMPTATION > REWARD > PUNISHMENT > SUCKER; 2*REWARD > TEMPTATION + SUCKER;
        # REWARD = 1; PUNISHMENT = 0; TEMPTATION > 1; -1 < SUCKER < 0
        sucker = np.round(np.random.uniform(-1, punishment-0.01, N), decimals=2)
        # temptation = np.round(np.random.uniform(reward+0.01, 2*reward-sucker-0.01, N), decimals=2)
        temptation = np.clip(np.random.uniform(reward-K+0.01, 2*reward-sucker-0.01, N), reward+0.01, 2*reward-sucker-0.01)
        temptation = np.round(temptation, decimals=2)
        assert np.sum(temptation > reward) == N and np.sum(reward > punishment) == N and np.sum(punishment > sucker) == N, f'{np.sum(temptation > reward)} and {np.sum(reward > punishment)} and {np.sum(punishment > sucker)}'
        assert np.sum(2*reward > temptation + sucker) == N, f'{np.sum(2*reward > temptation + sucker)}'
        return reward, temptation, sucker, punishment
    if name == 'SH':
        # stag hunt rule: REWARD > TEMPTATION > PUNISHMENT > SUCKER; TEMPTATION + SUCKER >= 2*PUNISHMENT;
        # REWARD = 1; PUNISHMENT = 0; -1 < SUCKER < 0; TEMPTATION < 1; -1 < REWARD - TEMPTATION + PUNISHMENT - SUCKER < 1
        diff = np.round(np.random.uniform(-1, 3, N), decimals=2)
        reward = np.maximum(reward+diff, reward)
        temptation = np.round(np.random.uniform(punishment+0.01, 1-0.01, N), decimals=2)
        temptation = np.minimum(temptation, 1-0.01)
        sucker = np.minimum(diff+temptation-reward-punishment, -0.01)
        # sucker = np.round(np.random.uniform(-1+0.5, punishment-0.01, N), decimals=2)
        # temptation = np.round(np.random.uniform(2*punishment-sucker, reward-0.01, N), decimals=2)
        assert np.sum(reward > temptation) == N and np.sum(temptation > punishment) == N and np.sum(punishment > sucker) == N, f'{np.sum(reward > temptation)} and {np.sum(temptation > punishment)} and {np.sum(punishment > sucker)}'
        return reward, temptation, sucker, punishment
    if name == 'SD':
        # snowdrift (chicken) rule: TEMPTATION > REWARD > SUCKER > PUNISHMENT;
        # REWARD = 1; PUNISHMENT = 0; SUCKER > 0; TEMPTATION > 1
        sucker = np.round(np.random.uniform(punishment + 0.01, reward - 0.01, N), decimals=2)
        temptation = np.round(np.random.uniform(reward + 0.01, 2*reward, N), decimals=2)
        assert np.sum(temptation > reward) == N and np.sum(reward > sucker) == N and np.sum(
            sucker > punishment) == N, f'{np.sum(temptation > reward)} and {np.sum(reward > sucker)} and {np.sum(sucker > punishment)}'
        return reward, temptation, sucker, punishment
    if name == 'Ha':
        # harmony rule: REWARD > TEMPTATION > SUCKER > PUNISHMENT;
        # REWARD = 1; PUNISHMENT = 0; SUCKER > 0; TEMPTATION < 1
        temptation = np.round(np.random.uniform(punishment + 0.02, reward - 0.01, N), decimals=2)
        sucker = np.round(np.random.uniform(punishment + 0.01, temptation-0.01, N), decimals=2)
        assert np.sum(reward > temptation) == N and np.sum(temptation > sucker) == N and np.sum(
            sucker > punishment) == N, f'{np.sum(reward > temptation)} and {np.sum(temptation > sucker)} and {np.sum(sucker > punishment)}'
        return reward, temptation, sucker, punishment

def generate_state(agent: object, h: int, n_actions: int, k: int = 1000, seed:int = 42):
    seed_everything(seed)
    # enumerate binary to generate h actions
    binary_enum = [i for i in range(n_actions ** h)]
    binary_list = []
    for i in binary_enum:
        x = [int(j) for j in list(bin(i).split('b')[1])]
        while len(x) < h:
            x.insert(0, 0)
        binary_list.append(torch.as_tensor(x))

    state_list = []

    if len(binary_list)**2 > 1000:
        for _ in range(k):
            h_actions = random.choices(binary_list, k=2)
            own_h_actions, opponent_h_actions = h_actions[0], h_actions[1]

            state = agent.state.state_repr(opponent_h_actions, own_h_actions)
            state = torch.permute(state.view(-1, h), (1, 0))  # important
            state_list.append(state)
    else:
        for i in binary_list:
            for j in binary_list:
                own_h_actions, opponent_h_actions = i, j
                state = agent.state.state_repr(opponent_h_actions, own_h_actions)
                state = torch.permute(state.view(-1, h), (1, 0))  # important
                state_list.append(state)
    return state_list