import argparse
import simulation
import matplotlib.pyplot as plt
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from main import Config
from utils import HiddenPrints
from agent.lstm_agent import LSTMAgent
from agent.actor_critic_agent import ActorCriticAgent
from agent.actor_critic_lstm_agent import ActorCriticLSTMAgent
from env import Environment

# --------------------------------------------------------------------------- #
# Parse command line arguments (CLAs):
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='Playground', prog='Prisoner dilemma simulation')
parser.add_argument('--name', default=None, type=str, help='Name of the run, will be in the output file')
parser.add_argument('--discount', default=0.95, type=float, help='Reward discount, range:(0,1]')
parser.add_argument('--n_episodes', default=10, type=int, help='Number of episodes within a batch')
parser.add_argument('--h', default=1, type=int, help='state amount')
parser.add_argument('--play_epsilon', default=1, type=float,
                    help='The greedy factor when each agent play the dilemma game')
parser.add_argument('--select_epsilon', default=0.1, type=float,
                    help='The greedy factor when each agent select the opponent')
parser.add_argument('--epsilon_decay', default=0.99, type=float, help='')
parser.add_argument('--min_epsilon', default=0.01, type=float, help='')
parser.add_argument('--reward', default=3, type=float, help='')
parser.add_argument('--temptation', default=5, type=float, help='')
parser.add_argument('--sucker', default=0, type=float, help='')
parser.add_argument('--punishment', default=1, type=float, help='')
parser.add_argument('--alpha', default=0.1, type=float, help='The alpha (learning rate) for RL learning')
parser.add_argument('--state_repr', default='unilabel',
                    choices=[None, 'uni', 'bi', 'unilabel', 'grudgerlabel', 'bireward'],
                    help='The state reprsentation method; (None: only use the opponent h actions; grudger: count mad)')
parser.add_argument('--batch_size', default=64, type=int, help='The bathc size for Neural Network')
parser.add_argument('--learning_rate', default=1e-3, help='The learning rate for optimizing Neural Network')
# --------------------------------------------------------------------------- #

N=100
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)

def configuration():
    global args
    args = parser.parse_args()
    config = {
        'reward': args.reward,
        'sucker': args.sucker,
        'temptation': args.temptation,
        'punishment': args.punishment,
        'n_episodes': args.n_episodes,
        'discount': args.discount,
        'play_epsilon': args.play_epsilon,
        'select_epsilon': args.select_epsilon,
        'epsilon_decay': args.epsilon_decay,
        'min_epsilon': args.min_epsilon,
        'alpha': args.alpha,
        'n_actions': 2,
        'h': args.h,
        'state_repr': args.state_repr,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
    }
    config = Config(config)
    print(config.__repr__)
    return config

def test_benchmark():
    config = configuration()
    mad = 2 if config.state_repr=='grudger' else 1
    l = torch.zeros(2**config.h*mad)
    result_dict = {0:l.clone(), 1:l.clone(), 2:l.clone(), 3:l.clone(), 4:l.clone(), 5:l.clone(), 6:l.clone()}
    strategies = {0:'ALLC',1:'ALLD',2:'TitForTat',3:'revTitForTat',4:'Random',5:'Grudger',6:'Pavlov'}
    discount = config.discount
    for n in tqdm(range(N)):
        with HiddenPrints():
            Q_list = simulation.benchmark(strategies, None, config)
        for idx,Q in enumerate(Q_list):
            result_dict[idx] += torch.argmax(Q, dim=1)
        config.discount = discount
    result_dict = {key: value / N for key, value in result_dict.items()}
    print(strategies)
    print(result_dict)

def test_lstm():
    strategies = {0: 'ALLC', 1: 'ALLD', 2: 'TitForTat', 3: 'revTitForTat', 4: 'Random', 5: 'Grudger', 6: 'Pavlov',
                  7: 'QLearning', 8: 'DQN', 10:'LSTMQN'}
    config = configuration()
    env = Environment(config)
    agent2 = simulation.constructOpponent(strategies[2], config)
    # agent2 = LSTMAgent('LSTMQN', config)
    # agent1 = ActorCriticAgent('A2C', config)
    agent1 = ActorCriticLSTMAgent('A2CLSTM', config)
    simulation.play(agent1, agent2, config.n_episodes, env)
    agent1.show()
    agent2.show()

    print(f'Your score: {agent1.running_score}\nOppo score: {agent2.running_score}')
    print(agent1.loss[:])
    plt.plot(agent1.loss)
    plt.title(f'agent1: {agent1.name}')
    plt.show()

if __name__ == "__main__":
    # test_benchmark()
    test_lstm()