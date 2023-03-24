import argparse
import simulation
import torch
# --------------------------------------------------------------------------- #
# Parse command line arguments (CLAs):
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='Playground', prog='Prisoner dilemma simulation')
parser.add_argument('--name', default=None, type=str, help='Name of the run, will be in the output file')
parser.add_argument('--discount', default=0.99, type=float, help='Reward discount or GAMMA, range:(0,1]')
parser.add_argument('--n_episodes', default=1000, type=int, help='Number of episodes within a batch')
parser.add_argument('--h', default=10, type=int, help='state amount')
parser.add_argument('--play_epsilon', default=1, type=float, help='The greedy factor when each agent play the dilemma game')
parser.add_argument('--select_epsilon', default=1, type=float, help='The greedy factor when each agent select the opponent')
parser.add_argument('--epsilon_decay', default=0.999, type=float, help='The decay coefficient of epsilon greedy policy of play_epsilon: (new_play_epsilon) = (old_play_epsilon)*epsilon_decay, play_epsilon >= min_epsilon')
parser.add_argument('--min_epsilon', default=0.01, type=float, help='The minimum epsilon value of play_epsilon')
parser.add_argument('--reward', default=3, type=float, help='The payoff when both agents cooperate')
parser.add_argument('--temptation', default=5, type=float, help='The payoff when you defect and your opponent cooperate')
parser.add_argument('--sucker', default=0, type=float, help='The payoff when both agents defect')
parser.add_argument('--punishment', default=1, type=float, help='The payoff when you cooperate and your opponent defects')
parser.add_argument('--alpha', default=0.1, type=float, help='The alpha (learning rate) for tabular q learning method')
parser.add_argument('--state_repr', default='bi-repr', type=str, choices=[None, 'uni', 'bi', 'unilabel', 'grudgerlabel', 'bi-repr'], help='The state reprsentation method; (None: only use the opponent h actions; grudger: count mad)')
parser.add_argument('--batch_size', default=64, type=int, help='The batch size for updating Neural Network-based RL method like dqn, lstm, a2c...')
parser.add_argument('--learning_rate', default=1e-3, type=float, help='The learning rate for optimizing Neural Network-based RL method like dqn, lstm, a2c...')
# parser.print_help()
# --------------------------------------------------------------------------- #

class Config():
    def __init__(self, config: dict):
        self.parse_config(**config)
    
    def parse_config(self, reward, sucker, temptation, punishment, n_episodes, discount, play_epsilon, select_epsilon, epsilon_decay, min_epsilon, alpha, n_actions, h, state_repr, batch_size, learning_rate):
        # game payoffs
        self.reward = reward
        self.sucker = sucker
        self.temptation = temptation
        self.punishment = punishment
        self.n_episodes = n_episodes
        self.discount = discount
        self.play_epsilon = play_epsilon
        self.select_epsilon = select_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.alpha = alpha
        self.n_actions = n_actions
        self.h = h
        self.state_repr = state_repr
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def __repr__(self):
        return 'Configs:\n' + ' episodes=' + str(self.n_episodes) + \
            ' discount=' + str(self.discount) + \
            '\npayoff matrix: ' + \
            ' r=' + str(self.reward) + \
            ' t=' + str(self.temptation) + \
            ' s=' + str(self.sucker) + \
            ' p=' + str(self.punishment) + \
            '\nplay_epsilon=' + str(self.play_epsilon) + \
            ' select_epsilon=' + str(self.select_epsilon) + \
            ' epsilon_decay=' + str(self.epsilon_decay) + \
            ' state_repr=' + str(self.state_repr) + \
            ' h=' + str(self.h)


def main():
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
    print(f'cuda = {torch.cuda.is_available()}')
    print('Here are your game options')
    print('press 0 to generate a benchmark against all strategies in geometric discount setting')
    print('press 1 to test an a strategy against all strategies')
    print('press 2 to play against a strategy of your choice ')
    print('press 3 to play a N agents game')
    choice = int(input())
    choices = {'0-alwaysCooperate','1-alwaysDefect','2-titForTat','3-reverseTitForTat','4-random','5-grudger','6-pavlov','7-qLearning','8-lstm-TFT','9-dqn','10-lstmqn','11-a2c','12-a2c-lstm'}
    rl_choices = {'7-qLearning','8-lstm-pavlov','9-dqn','10-lstmqn','11-a2c','12-a2c-lstm'}
    strategies = {0:'ALLC',1:'ALLD',2:'TitForTat',3:'revTitForTat',4:'Random',5:'Grudger',6:'Pavlov',7:'QLearning',8:'LSTM',9:'DQN',10:'LSTMQN',11:'A2C',12:'A2CLSTM'}

    if choice == 0:
        # print('here are the strategies, choose one\n', choices)
        # num = int(input('choose a strategy via number '))
        # print('You will use the strategy ' + strategies[num])
        simulation.benchmark(strategies, None, config)

    if choice == 1:
        print('here are the strategies, choose one\n', choices)
        num = int(input('choose a strategy via number '))
        simulation.twoSimulate(strategies, num, config)

    if choice == 2:
        print('right now you are a rl agent, choice one strategy')
        print(rl_choices)
        rl_num = int(input('choose a strategy via number '))
        print('who do you want to play against')
        print(choices)
        num = int(input('choose a strategy via number '))
        # rounds = int(input('how many rounds do you want to play:'))
        # if rounds > config.n_episodes:
        #     config.n_episodes = rounds
        simulation.twoSimulate(dict({num: strategies[num], rl_num: strategies[rl_num]}), rl_num, config)

    if choice == 3:
        simulation.multiAgentSimulate(strategies, config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
if __name__ == "__main__":
    main()