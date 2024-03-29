import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from agent.abstract_agent import AbstractAgent
from model import NeuralNetwork, DQN
from utils import *
from component import Memory, ReplayBuffer

MAD_THRESHOLD = 5
TARGET_UPDATE = 10
HIDDEN_SIZE = 128
BUFFER_SIZE = 1000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DQNAgent(AbstractAgent):
    # h is every agents' most recent h actions are visiable to others which is composed to state
    def __init__(self, name: str, config: object):
        """

        Parameters
        ----------
        config : object
        name: str = DQN
        """
        super(DQNAgent, self).__init__(config)
        self.name = name
        self.own_memory = torch.zeros((config.n_episodes*10000, ))
        self.opponent_memory = torch.zeros((config.n_episodes*10000, ))
        self.state = self.StateRepr(method=config.state_repr)
        self.build()
        self.policy = self.EpsilonPolicy(self.policy_net, config.play_epsilon, config.n_actions)  # an object
        self.memory = ReplayBuffer(BUFFER_SIZE)  # an object
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.config.learning_rate)
        self.criterion = nn.SmoothL1Loss()
        self.loss = []
        self.model = DQN()

    def build(self):
        """State, Policy, Memory, Q are objects"""
        input_size = self.config.h if self.config.state_repr=='uni' else self.config.h*2 if 'bi' in self.config.state_repr else 1
        self.policy_net = NeuralNetwork(input_size, self.config.n_actions, HIDDEN_SIZE).to(device) if self.name=='DQN' else None # an object
        self.target_net = NeuralNetwork(input_size, self.config.n_actions, HIDDEN_SIZE).to(device) if self.name=='DQN' else None # an object
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print(self.target_net.eval())
        

    def act(self, oppo_agent: object):
        """
        Agent act based on the oppo_agent's information
        Parameters
        ----------
        oppo_agent: object

        Returns
        -------
        action index
        """
        # get opponent's last h move
        opponent_h_actions = torch.as_tensor(
            oppo_agent.own_memory[oppo_agent.play_times - self.config.h: oppo_agent.play_times])
        own_h_actions = torch.as_tensor(
            self.own_memory[self.play_times - self.config.h: self.play_times])
        if self.play_times >= self.config.h and oppo_agent.play_times >= self.config.h:
            self.state.state = self.state.state_repr(opponent_h_actions, own_h_actions)
        else:
            self.state.state = None
        return int(self.__select_action())

    def __select_action(self):
        """ selection action based on epsilon greedy policy """
        a = self.policy.sample_action(self.state.state)
        return a

    def update(self, reward: float, own_action: int, opponent_action: int):
        super(DQNAgent, self).update(reward)
        self.own_memory[self.play_times - 1] = own_action
        self.opponent_memory[self.play_times - 1] = opponent_action
        # self.state.oppo_memory = self.opponent_memory[:self.play_times]

    def get_next_state(self,  oppo_agent: object, state: Type.TensorType = None):
        # get next state
        opponent_h_actions = torch.as_tensor(
            oppo_agent.own_memory[oppo_agent.play_times - self.config.h: oppo_agent.play_times])
        own_h_actions = torch.as_tensor(
            self.own_memory[self.play_times - self.config.h: self.play_times])
        self.state.next_state = self.state.state_repr(opponent_h_actions, own_h_actions)
        self.state.state = self.state.state if state is None else state

    def optimize(self, action: int, reward: float, oppo_agent: object, state: Type.TensorType = None, flag: bool=True):
        super(DQNAgent, self).optimize(action, reward, oppo_agent)
        if self.state.state is None:
            return None

        self.get_next_state(oppo_agent)
        # push the transition into ReplayBuffer
        self.memory.push(self.state.state, action, self.state.next_state, reward)

        if flag:
            self.__optimize_model()
            # Update the target network, copying all weights and biases in DQN
            if self.play_times % TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_batch(self, transitions):
        # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
        state, action, next_state, reward = zip(*transitions)

        # convert to PyTorch and define types
        state = torch.stack(list(state), dim=0).to(device)
        action = torch.tensor(action, dtype=torch.int64, device=device)[:, None]  # Need 64 bit to use them as index
        next_state = torch.stack(list(next_state), dim=0).to(device)
        reward = torch.tensor(reward, dtype=torch.float, device=device)[:, None]
        # loss is measured from error between current and newly expected Q values
        batch = Memory.Transition(state, action, next_state, reward)
        return batch

    def __optimize_model(self):
        """ Train and optimize our model """
        # don't learn without some decent experience
        if len(self.memory.memory) < self.config.batch_size:
            return None
        # random transition batch is taken from experience replay memory
        transitions = self.memory.sample(self.config.batch_size)
        batch = self.get_batch(transitions)

        self.model.train(self, batch)
        # test
        # print(f'==================={self.play_times}===================')
        # print(f'transition: \n{np.hstack((state.numpy(),action.numpy(),next_state.numpy(),reward.numpy()))}')
        # print(f'transition: \nstate: {np.squeeze(state.numpy())}\naction: {np.squeeze(action.numpy())}\nnext_s: {np.squeeze(next_state.numpy())}\nreward: {np.squeeze(reward.numpy())}')
        # print(f'loss: {loss.item()}')

    def determine_convergence(self, threshold: int, k: int):
        if self.play_times < 2 * k:
            return False
        history_1 = self.own_memory[self.play_times - k: self.play_times]
        history_2 = self.own_memory[self.play_times - 2 * k: self.play_times - k]
        difference = torch.sum(torch.abs(history_1 - history_2))
        if difference > threshold:
            return False
        else:
            return True

    def show(self):
        print("==================================================")
        print(f'Your action: {self.own_memory[self.play_times-20:self.play_times]}\nOppo action: {self.opponent_memory[self.play_times-20:self.play_times]}')





