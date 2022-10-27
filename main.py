import argparse
import simulation
import torch
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
parser.add_argument('--alpha', default=0.5, type=float, help='The alpha for Q_learning')
# parser.print_help()
# --------------------------------------------------------------------------- #

class Config():
    def __init__(self, config):
        self.parse_config(**config)
    
    def parse_config(self, reward, sucker, temptation, punishment, n_episodes, discount, play_epsilon, select_epsilon, epsilon_decay, min_epsilon, alpha, n_actions, h):
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

    def __repr__(self):
        return 'Configs: ' + ' episodes=' + str(self.n_episodes) + \
            ' discount=' + str(self.discount) + \
            '\npayoff matrix: ' + \
            ' r=' + str(self.reward) + \
            ' t=' + str(self.temptation) + \
            ' s=' + str(self.sucker) + \
            ' p=' + str(self.punishment) + \
            ' play_epsilon=' + str(self.play_epsilon) + \
            ' select_epsilon=' + str(self.select_epsilon) + \
            ' epsilon_decay=' + str(self.epsilon_decay)


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
    }
    config = Config(config)
    print(config.__repr__)
    print('Here are your game options')
    print('press 0 to test an a strategy against all strategies in geometric discount setting')
    print('press 1 to test an a strategy against all strategies')
    print('press 2 to play against a strategy of your choice ')
    print('press 3 to play a N agents game')
    choice = int(input())
    choices = {'0-alwaysCooperate','1-alwaysDefect','2-titForTat','3-reverseTitForTat','4-random','5-grudger','6-pavlov','7-qLearning','8-mc','9-dqn'}
    rl_choices = {'7-qLearning','8-mc','9-dqn'}
    strategies = {0:'ALLC',1:'ALLD',2:'TitForTat',3:'revTitForTat',4:'Random',5:'Grudger',6:'Pavlov',7:'QLearning',8:'MCLearning'}

    if choice == 0:
        print('here are the strategies, choose one\n', choices)
        num = int(input('choose a strategy via number '))
        print('You will use the strategy ' + strategies[num])
        simulation.testTransition(strategies, num, config)

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
        print('how many agents to play')
        n_agents = int(input())
        simulation.multiSimulate(n_agents, strategies, config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
if __name__ == "__main__":
    main()