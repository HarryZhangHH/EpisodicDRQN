import torch.nn as nn
from agent.abstract_agent import AbstractAgent
from model import LSTM, LSTMVariant, DQN
from utils import *
from component import Memory, ReplayBuffer
import sys

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TARGET_UPDATE = 10
HIDDEN_SIZE = 128
FEATURE_SIZE = 4
BUFFER_SIZE = 1000
NUM_LAYER = 1

class LSTMAgent(AbstractAgent):
    # h is every agents' most recent h actions are visiable to others which is composed to state
    def __init__(self, name: str, config: object):
        """
        Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            Network (str): dqn network type
            layer_size (int): size of the hidden layer
            BATCH_SIZE (int): size of the training batch
            BUFFER_SIZE (int): size of the replay memory
            LR (float): learning rate
            TAU (float): tau for soft updating the network weights
            GAMMA (float): discount factor
            UPDATE_EVERY (int): update frequency
            device (str): device that is used for the compute
            seed (int): random seed
        """
        super(LSTMAgent, self).__init__(config)
        self.name = name
        self.own_memory = torch.zeros((config.n_episodes*1000, ))
        self.opponent_memory = torch.zeros((config.n_episodes*1000, ))
        self.state = self.StateRepr(method=config.state_repr)
        self.vaiant_flag = True if 'repr' in config.state_repr else False
        self.feature_size = FEATURE_SIZE
        self.build() if not self.vaiant_flag else self.build2()
        self.policy = self.EpsilonPolicy(self.policy_net, config.play_epsilon, self.n_actions)  # an object
        self.memory = ReplayBuffer(BUFFER_SIZE)  # an object
        self.loss = []
        self.criterion = nn.CrossEntropyLoss() if self.name=='LSTM' else nn.SmoothL1Loss()
        self.model = SupervisedLearning() if self.name=='LSTM' else DQN()

    def build(self):
        """ Build a normal LSTM network 
        PolicyNet and TargetNet are objects """
        input_size = 2 if self.config.state_repr == 'bi' else 1
        self.policy_net = LSTM(input_size, HIDDEN_SIZE, NUM_LAYER, self.n_actions).to(device)
        self.target_net = LSTM(input_size, HIDDEN_SIZE, NUM_LAYER, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.config.learning_rate)
        print(self.target_net.eval())
    
    def build2(self):
        """ Build a variant LSTM network, an ensemble LSTM and DENSE Network """
        input_size = 2 if 'bi' in self.config.state_repr else 1
        self.policy_net = LSTMVariant(input_size, HIDDEN_SIZE, NUM_LAYER, self.feature_size, self.n_actions, int(HIDDEN_SIZE/2)).to(device)
        self.target_net = LSTMVariant(input_size, HIDDEN_SIZE, NUM_LAYER, self.feature_size, self.n_actions, int(HIDDEN_SIZE/2)).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.config.learning_rate)
        print(self.target_net.eval())

        
    def act(self, oppo_agent: object):
        """
        Agent act based on the oppo_agent's information
        Parameters
        ----------
        oppo_agent: object

        Returns
        -------
        action index: int
        """
        # get opponent's last h move
        opponent_h_actions = torch.as_tensor(
            oppo_agent.own_memory[oppo_agent.play_times - self.config.h: oppo_agent.play_times])
        own_h_actions = torch.as_tensor(
            self.own_memory[self.play_times - self.config.h: self.play_times])
        
        if self.play_times >= self.config.h and oppo_agent.play_times >= self.config.h:
            self.state.state = self.state.state_repr(opponent_h_actions, own_h_actions)
            self.state.state = torch.permute(self.state.state.view(-1, self.config.h), (1, 0)) # important

            if self.vaiant_flag:
                feature = self.generate_feature(oppo_agent)
                self.state.state = (self.state.state, feature)
        else:
            self.state.state = None
        return int(self.__select_action())

    def __select_action(self):
        """selection action based on epsilon greedy policy """
        a = self.policy.sample_action(self.state.state)
        return a

    def generate_feature(self, oppo_agent: object):
        """ 
        Generate extra features 
        own_reward_ratio, oppo_reward_ratio, own_defect_ratio, oppo_defect_ratio
        """
        max_reward = self.config.temptation/(1-self.config.discount)
        own_reward = self.running_score
        oppo_reward = oppo_agent.running_score
        own_defect_ratio = calculate_sum(self.own_memory)/self.play_times
        own_faced_defect_ratio = calculate_sum(self.opponent_memory)/self.play_times
        oppo_defect_ratio = calculate_sum(oppo_agent.own_memory)/oppo_agent.play_times
        oppo_faced_defect_ratio = calculate_sum(oppo_agent.opponent_memory)/oppo_agent.play_times
        own_reward_ratio = own_reward/max_reward
        oppo_reward_ratio = oppo_reward/max_reward
        own_play_times_ratio = min(1, self.play_times/(self.config.n_episodes*5))
        oppo_play_times_ratio = min(1, oppo_agent.play_times/(self.config.n_episodes*5))
        if self.feature_size == 4:
            return torch.FloatTensor([own_reward_ratio, own_defect_ratio, oppo_reward_ratio, oppo_defect_ratio])
        else:
            return torch.FloatTensor([own_reward_ratio, own_defect_ratio, own_faced_defect_ratio, own_play_times_ratio,
                                      oppo_reward_ratio, oppo_defect_ratio, oppo_faced_defect_ratio, oppo_play_times_ratio])

    def update(self, reward: float, own_action: int, opponent_action: int):
        super(LSTMAgent, self).update(reward)
        self.own_memory[self.play_times - 1] = own_action
        self.opponent_memory[self.play_times - 1] = opponent_action
        # self.state.oppo_memory = self.opponent_memory[:self.play_times]

    def get_next_state(self,  oppo_agent: object, state: Type.TensorType = None):
        # get opponent's last h move
        opponent_h_actions = torch.as_tensor(
            oppo_agent.own_memory[oppo_agent.play_times - self.config.h: oppo_agent.play_times])
        own_h_actions = torch.as_tensor(
            self.own_memory[self.play_times - self.config.h: self.play_times])
        self.state.next_state = self.state.state_repr(opponent_h_actions, own_h_actions)
        self.state.next_state = torch.permute(self.state.next_state.view(-1, self.config.h), (1, 0))  # important
        if self.vaiant_flag:
            feature = self.generate_feature(oppo_agent)
            self.state.next_state = (self.state.next_state.numpy(), feature.numpy())
            self.state.state = (self.state.state[0].numpy(), self.state.state[1].numpy()) if state is None else (state[0].numpy(), state[1].numpy())

    def optimize(self, action: int, reward: float, oppo_agent: object, state: Type.TensorType = None, flag: bool = True):
        """ push the trajectoriy into the ReplayBuffer and optimize the model """
        super(LSTMAgent, self).optimize(action, reward, oppo_agent)
        if self.state.state is None:
            return None

        # get opponent's last h move
        self.get_next_state(oppo_agent, state)

        # push the transition into ReplayBuffer
        if self.name == 'LSTM':
            self.memory.push(self.state.state, action, int(oppo_agent.own_memory[oppo_agent.play_times - 1]), reward)
        if self.name == 'LSTMQN':
            self.memory.push(self.state.state, action, self.state.next_state, reward)

        if flag:
            # print(f'Episode {self.play_times}:  {own_action}, {opponent_action}') if self.play_times > self.config.batch_size else None
            self.__optimize_model()
            self.policy.update_epsilon(self.config)
            # Update the target network, copying all weights and biases in DQN
            if self.play_times % TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_batch(self, transitions):
        # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
        state, action, next_state, reward = zip(*transitions)
        # convert to PyTorch and define types
        if self.name == 'LSTM':
            if self.vaiant_flag:
                h_action = torch.from_numpy(np.vstack(np.array(state, dtype=object)[:, 0]).astype(np.float)).to(
                    device).view(self.config.batch_size, self.config.h, -1)
                features = torch.from_numpy(np.vstack(np.array(state, dtype=object)[:, 1]).astype(np.float)).to(
                    device).view(self.config.batch_size, self.feature_size)
                state = (h_action, features)
            else:
                state = torch.stack(list(state), dim=0).view(self.config.batch_size, self.config.h, -1).to(device)

        elif self.name == 'LSTMQN':
            if self.vaiant_flag:
                # test
                # print(np.array(state, dtype=object)[:,0])
                # print(np.vstack(np.array(state, dtype=object)[:,0]))
                # print( torch.from_numpy(np.vstack(np.array(state, dtype=object)[:,0]).astype(np.float64)).view(self.config.batch_size, self.config.h, -1))
                # print( torch.from_numpy(np.vstack(np.array(state, dtype=object)[:,1]).astype(np.float64)).size())
                # sys.exit()
                h_action = torch.from_numpy(np.vstack(np.array(state, dtype=object)[:, 0]).astype(np.float)).view(
                    self.config.batch_size, self.config.h, -1).to(device)
                features = torch.from_numpy(np.vstack(np.array(state, dtype=object)[:, 1]).astype(np.float)).view(
                    self.config.batch_size, self.feature_size).to(device)
                state = (h_action, features)
                h_action = torch.from_numpy(np.vstack(np.array(next_state, dtype=object)[:, 0]).astype(np.float)).view(
                    self.config.batch_size, self.config.h, -1).to(device)
                features = torch.from_numpy(np.vstack(np.array(next_state, dtype=object)[:, 1]).astype(np.float)).view(
                    self.config.batch_size, self.feature_size).to(device)
                next_state = (h_action, features)
            else:
                state = torch.stack(list(state), dim=0).view(self.config.batch_size, self.config.h, -1).to(device)
                next_state = torch.stack(list(next_state), dim=0).view(self.config.batch_size, self.config.h, -1).to(
                    device)
            action = torch.tensor(action, dtype=torch.int64, device=device)[:, None]  # Need 64 bit to use them as index
            reward = torch.tensor(reward, dtype=torch.float, device=device)[:, None]

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

class SupervisedLearning(object):
    @staticmethod
    def train(agent: object, batch: object):
        target = torch.tensor(batch.next_state, dtype=torch.int64, device=device)
        outputs = agent.policy_net(batch.state)
        loss = agent.criterion(outputs, target)

        # backpropagation of loss to Neural Network (PyTorch magic)
        agent.optimizer.zero_grad()
        loss.backward()
        for param in agent.policy_net.parameters():
            param.grad.data.clamp_(-1,
                                   1)  # DQN gradient clipping: Clamps all elements in input into the range [ min, max ].
        agent.optimizer.step()
        agent.loss.append(loss.item())
