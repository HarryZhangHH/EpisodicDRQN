import torch.nn as nn
from agent.abstract_agent import AbstractAgent
from model import LSTM
from utils import *

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TARGET_UPDATE = 10
HIDDEN_SIZE = 128

class LSTMAgent(AbstractAgent):
    # h is every agents' most recent h actions are visiable to others which is composed to state
    def __init__(self, name, config):
        """

        Parameters
        ----------
        config
        name = LSTM or LSTMQN
        """
        super(LSTMAgent, self).__init__(config)
        self.name = name
        self.own_memory = torch.zeros((config.n_episodes*1000, ))
        self.opponent_memory = torch.zeros((config.n_episodes*1000, ))
        self.play_epsilon = config.play_epsilon
        self.State = self.StateRepr(method=config.state_repr)
        self.build()
        self.loss = []

    def build(self):
        """State, Policy, Memory, Q are objects"""
        input_size = 2 if self.config.state_repr == 'bi' else 1
        self.PolicyNet = LSTM(input_size, HIDDEN_SIZE, 1, self.config.n_actions).to(device)
        self.TargetNet = LSTM(input_size, HIDDEN_SIZE, 1, self.config.n_actions).to(device)
        self.TargetNet.load_state_dict(self.PolicyNet.state_dict())
        print(self.PolicyNet.eval())
        self.Policy = self.EpsilonPolicy(self.PolicyNet, self.play_epsilon, self.config.n_actions)  # an object
        self.Memory = self.ReplayBuffer(1000)  # an object
        self.Optimizer = torch.optim.Adam(self.PolicyNet.parameters(), lr=self.config.learning_rate)

    def act(self, oppo_agent):
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
        self.opponent_action = torch.as_tensor(
            oppo_agent.own_memory[oppo_agent.play_times - self.config.h: oppo_agent.play_times])
        self.own_action = torch.as_tensor(
            self.own_memory[self.play_times - self.config.h: self.play_times])

        if self.play_times >= self.config.h:
            self.State.state = self.State.state_repr(self.opponent_action, self.own_action)
        return int(self.select_action())

    def select_action(self):
        # selection action based on epsilon greedy policy
        self.State.state = torch.permute(self.State.state.view(-1, self.config.h), (1, 0)) if self.State.state is not None else None # important
        a = self.Policy.sample_action(self.State.state)

        # epsilon decay
        if self.play_epsilon > self.config.min_epsilon:
            self.play_epsilon *= self.config.epsilon_decay
        self.Policy.set_epsilon(self.play_epsilon)
        return a

    def update(self, reward, own_action, opponent_action):
        super(LSTMAgent, self).update(reward)
        self.own_memory[self.play_times - 1] = own_action
        self.opponent_memory[self.play_times - 1] = opponent_action
        self.State.oppo_memory = self.opponent_memory[:self.play_times]

        if self.State.state is not None:
            self.State.next_state = self.State.state_repr(torch.cat([self.opponent_action[1:], torch.as_tensor([opponent_action])]),
                                                          torch.cat([self.own_action[1:], torch.as_tensor([own_action])]))
            # print(self.State.next_state)
            self.State.next_state = torch.permute(self.State.next_state.view(-1, self.config.h), (1, 0))  # important
            # push the transition into ReplayBuffer
            if self.name == 'LSTM':
                self.Memory.push(self.State.state, own_action, opponent_action, reward)
            elif self.name == 'LSTMQN':
                self.Memory.push(self.State.state, own_action, self.State.next_state, reward)
            # print(f'Episode {self.play_times}:  {own_action}, {opponent_action}') if self.play_times > self.config.batch_size else None
            self.optimize_model()
            # Update the target network, copying all weights and biases in DQN
            if self.play_times % TARGET_UPDATE == 0:
                self.TargetNet.load_state_dict(self.PolicyNet.state_dict())


    def optimize_model(self):
        """ Train our model """
        def compute_q_vals(Q, states, actions):
            """
            This method returns Q values for given state action pairs.

            Args:
                Q: Q-net  (object)
                states: a tensor of states. Shape: batch_size x obs_dim
                actions: a tensor of actions. Shape: Shape: batch_size x 1
            Returns:
                A torch tensor filled with Q values. Shape: batch_size x 1.
            """
            return torch.gather(Q(states), 1, actions)
        def compute_targets(Q, rewards, next_states, discount_factor):
            """
            This method returns targets (values towards which Q-values should move).

            Args:
                Q: Q-net  (object)
                rewards: a tensor of rewards. Shape: Shape: batch_size x 1
                next_states: a tensor of states. Shape: batch_size x obs_dim
                discount_factor: discount
            Returns:
                A torch tensor filled with target values. Shape: batch_size x 1.
            """
            return rewards + discount_factor * torch.max(Q(next_states), 1)[0].reshape((-1, 1))

        # don't learn without some decent experience
        if len(self.Memory.memory) < self.config.batch_size:
            return None
        # random transition batch is taken from experience replay memory
        transitions = self.Memory.sample(self.config.batch_size)
        # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
        state, action, next_state, reward = zip(*transitions)
        # convert to PyTorch and define types
        state = torch.stack(list(state), dim=0).to(device).view(self.config.batch_size, self.config.h, -1)
        if self.name == 'LSTM':
            target = torch.tensor(next_state, dtype=torch.int64, device=device)
            outputs = self.PolicyNet(state)
            criterion = nn.CrossEntropyLoss()

        elif self.name == 'LSTMQN':
            next_state = torch.stack(list(next_state), dim=0).to(device).view(self.config.batch_size, self.config.h, -1)
            action = torch.tensor(action, dtype=torch.int64, device=device)[:,None]  # Need 64 bit to use them as index
            reward = torch.tensor(reward, dtype=torch.float, device=device)[:,None]
            criterion = nn.SmoothL1Loss()
            # compute the q value
            outputs = compute_q_vals(self.PolicyNet, state, action)
            with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
                target = compute_targets(self.TargetNet, reward, next_state, self.config.discount)

        # loss is measured from error between current and newly expected Q values
        loss = criterion(outputs, target)
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

    def show(self):
        print("==================================================")
        print(f'Your action: {self.own_memory[self.play_times-20:self.play_times]}\nOppo action: {self.opponent_memory[self.play_times-20:self.play_times]}')

