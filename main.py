import argparse
import simulation
import torch
# --------------------------------------------------------------------------- #
# Parse command line arguments (CLAs):
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='Playground', prog='Prisoner dilemma simulation')
parser.add_argument('--name', default=None, type=str, help='Name of the run, will be in the output file')
parser.add_argument('--discount', default=0.95, type=float, help='Reward discount')
parser.add_argument('--n_episodes', default=10, type=int, help='Number of episodes within a batch')
parser.add_argument('--h', default=1, type=int, help='state amount')
parser.add_argument('--play_epsilon', default=0.1, type=float, help='The greedy factor when each agent play the dilemma game')
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
    
    def parse_config(self, reward, sucker, temptation, punishment, n_episodes, discount, play_epsilon, alpha, n_actions, h):
        # game payoffs
        self.reward = reward
        self.sucker = sucker
        self.temptation = temptation
        self.punishment = punishment
        self.n_episodes = n_episodes
        self.discount = discount
        self.play_epsilon = play_epsilon
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
            ' epsilon=' + str(self.play_epsilon)


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
        'alpha': args.alpha,
        'n_actions': 2,
        'h': args.h,
    }
    config = Config(config)
    print(config.__repr__)

    print('Here are your game options')
    print('press 1 to test an a strategy against all strategies')
    print('press 2 to play against a strategy of your choice ')
    choice = int(input())
    choices = {'1-alwaysCooperate','2-alwaysDefect','3-titForTat','4-random','5-grudger','6-qlearning'}
    strategies = {1:'ALLC',2:'ALLD',3:'TitForTat',4:'Random',5:'Grudger'}

    if choice == 1:
        print('here are the strategies, choose one\n', choices)
        num = int(input('choose a strategy via number '))
        if num < 6:
            print('You will use the strategy '+strategies[num])
            simulation.testStrategy(strategies, num, config)
        else:
            print('You will use the strategy q-learning')
            simulation.rlSimulate(strategies, config)

    if choice == 2:
        print('right now you are a q-learning agent, who do you want to play against')
        print(choices)
        num = int(input('choose a strategy via number'))
        strategy = strategies[num]
        # rounds = int(input('how many rounds do you want to play:'))
        # if rounds > config.n_episodes:
        #     config.n_episodes = rounds
        simulation.rlSimulate(strategy, config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
if __name__ == "__main__":
    main()