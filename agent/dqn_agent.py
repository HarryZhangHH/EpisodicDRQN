import torch.nn.functional as F
import matplotlib.pyplot as plt
from agent.abstract_agent import AbstractAgent
from model import NeuralNetwork
from utils import *

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
        self.State = self.StateRepr(method=config.state_repr)
        self.build()
        self.Policy = self.EpsilonPolicy(self.PolicyNet, config.play_epsilon, config.n_actions)  # an object
        self.Memory = self.ReplayBuffer(BUFFER_SIZE)  # an object
        self.Optimizer = torch.optim.Adam(self.PolicyNet.parameters(), lr=self.config.learning_rate)
        self.loss = []

    def build(self):
        """State, Policy, Memory, Q are objects"""
        input_size = self.config.h if self.config.state_repr=='uni' else self.config.h*2 if 'bi' in self.config.state_repr else 1
        self.PolicyNet = NeuralNetwork(input_size, self.config.n_actions, HIDDEN_SIZE).to(device) if self.name=='DQN' else None # an object
        self.TargetNet = NeuralNetwork(input_size, self.config.n_actions, HIDDEN_SIZE).to(device) if self.name=='DQN' else None # an object
        self.TargetNet.load_state_dict(self.PolicyNet.state_dict())
        print(self.TargetNet.eval())
        

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
            self.State.state = self.State.state_repr(opponent_h_actions, own_h_actions)
        else:
            self.State.state = None
        return int(self.__select_action())

    def __select_action(self):
        """ selection action based on epsilon greedy policy """
        a = self.Policy.sample_action(self.State.state)
        return a

    def update(self, reward: float, own_action: int, opponent_action: int):
        super(DQNAgent, self).update(reward)
        self.own_memory[self.play_times - 1] = own_action
        self.opponent_memory[self.play_times - 1] = opponent_action
        # self.State.oppo_memory = self.opponent_memory[:self.play_times]

    def optimize(self, action: int, reward: float, oppo_agent: object, state: Type.TensorType = None):
        super(DQNAgent, self).optimize(action, reward, oppo_agent)
        if self.State.state is None:
            return None

        # get next state
        opponent_h_actions = torch.as_tensor(
            oppo_agent.own_memory[oppo_agent.play_times - self.config.h: oppo_agent.play_times])
        own_h_actions = torch.as_tensor(
            self.own_memory[self.play_times - self.config.h: self.play_times])
        self.State.next_state = self.State.state_repr(opponent_h_actions, own_h_actions)
        self.State.state = self.State.state if state is None else state

        # push the transition into ReplayBuffer
        self.Memory.push(self.State.state, action, self.State.next_state, reward)
        self.__optimize_model()
        # Update the target network, copying all weights and biases in DQN
        if self.play_times % TARGET_UPDATE == 0:
            self.TargetNet.load_state_dict(self.PolicyNet.state_dict())

    def __optimize_model(self):
        """ Train our model """
        # don't learn without some decent experience
        if len(self.Memory.memory) < self.config.batch_size:
            return None
        # random transition batch is taken from experience replay memory
        transitions = self.Memory.sample(self.config.batch_size)
        # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
        state, action, next_state, reward = zip(*transitions)
        # convert to PyTorch and define types
        state = torch.stack(list(state), dim=0).to(device)
        action = torch.tensor(action, dtype=torch.int64, device=device)[:, None]  # Need 64 bit to use them as index
        next_state = torch.stack(list(next_state), dim=0).to(device)
        reward = torch.tensor(reward, dtype=torch.float, device=device)[:, None]
        # compute the q value
        q_val = compute_q_vals(self.PolicyNet, state, action)
        with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
            target = compute_targets(self.TargetNet, reward, next_state, self.config.discount)

        # loss is measured from error between current and newly expected Q values
        loss = F.smooth_l1_loss(q_val, target)
        # backpropagation of loss to Neural Network (PyTorch magic)
        self.Optimizer.zero_grad()
        loss.backward()
        for param in self.PolicyNet.parameters():
            param.grad.data.clamp_(-1, 1)  # DQN gradient clipping: Clamps all elements in input into the range [ min, max ].
        self.Optimizer.step()
        self.loss.append(loss.item())
        # test
        # print(f'==================={self.play_times}===================')
        # print(f'transition: \n{np.hstack((state.numpy(),action.numpy(),next_state.numpy(),reward.numpy()))}')
        # print(f'transition: \nstate: {np.squeeze(state.numpy())}\naction: {np.squeeze(action.numpy())}\nnext_s: {np.squeeze(next_state.numpy())}\nreward: {np.squeeze(reward.numpy())}')
        # print(f'loss: {loss.item()}')

    def determine_convergence(self, threshold: int, k: int):
        return super(DQNAgent, self).determine_convergence(threshold, k)

    def show(self):
        print("==================================================")
        print(f'Your action: {self.own_memory[self.play_times-20:self.play_times]}\nOppo action: {self.opponent_memory[self.play_times-20:self.play_times]}')





