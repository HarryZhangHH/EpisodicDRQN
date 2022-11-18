from main import Config
import argparse
import simulation
import torch
from utils import HiddenPrints
from tqdm import tqdm
# --------------------------------------------------------------------------- #
# Parse command line arguments (CLAs):
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='Playground', prog='Prisoner dilemma simulation')
parser.add_argument('--name', default=None, type=str, help='Name of the run, will be in the output file')
parser.add_argument('--discount', default=0.95, type=float, help='Reward discount, range:(0,1]')
parser.add_argument('--n_episodes', default=10, type=int, help='Number of episodes within a batch')
parser.add_argument('--h', default=1, type=int, help='state amount')
parser.add_argument('--play_epsilon', default=1, type=float, help='The greedy factor when each agent play the dilemma game')
parser.add_argument('--select_epsilon', default=0.1, type=float, help='The greedy factor when each agent select the opponent')
parser.add_argument('--epsilon_decay', default=0.99, type=float, help='')
parser.add_argument('--min_epsilon', default=0.01, type=float, help='')
parser.add_argument('--reward', default=3, type=float, help='')
parser.add_argument('--temptation', default=5, type=float, help='')
parser.add_argument('--sucker', default=0, type=float, help='')
parser.add_argument('--punishment', default=1, type=float, help='')
parser.add_argument('--alpha', default=0.1, type=float, help='The alpha (learning rate) for RL learning')
parser.add_argument('--state_repr', default=None, choices=[None, 'grudger'], help='The state reprsentation method; (None: only use the opponent h actions; grudger: count mad)')
parser.add_argument('--batch_size', default=64, help='The bathc size for Neural Network')
# --------------------------------------------------------------------------- #
N=100

def test_benchmark():
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
    }
    config = Config(config)
    print(config.__repr__)
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

if __name__ == "__main__":
    test_benchmark()