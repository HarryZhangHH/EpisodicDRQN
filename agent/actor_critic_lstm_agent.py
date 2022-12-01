import torch.nn.functional as F
from env import Environment
from agent.abstract_agent import AbstractAgent
from agent.fix_strategy_agent import StrategyAgent
from model import A2CLSTM
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HIDDEN_SIZE = 128
TARGET_UPDATE = 10
ENTROPY_COEF = 0.01
CRITIC_COEF = 0.5
WORKER_NUM = 16

class ActorCriticLSTMAgent(AbstractAgent):
    # h is every agents' most recent h actions are visiable to others which is composed to state
    def __init__(self, name, config):
        """

        Parameters
        ----------
        config
        name = A2CLSTM
        """
        super(ActorCriticLSTMAgent, self).__init__(config)
        self.name = name
        self.own_memory = torch.zeros((config.n_episodes*1000, ))
        self.opponent_memory = torch.zeros((config.n_episodes*1000, ))
        self.State = self.StateRepr(method=config.state_repr)
        self.build()
        self.loss = []

    def build(self):
        """State, Policy, Memory, Q are objects"""
        self.input_size = 2 if self.config.state_repr == 'bi' else 1
        self.PolicyNet = A2CLSTM(self.input_size, HIDDEN_SIZE, 1, self.config.n_actions).to(device)  # an object

        if 'Worker' not in self.name:
            self.Memory = self.ReplayBuffer(100)  # an object
            self.Optimizer = torch.optim.Adam(self.PolicyNet.parameters(), lr=self.config.learning_rate)
            self.Workers = Worker(self.config)

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
        a = self.PolicyNet.act(self.State.state[None]) if self.State.state is not None else random.randint(0, self.config.n_actions-1)
        # self.PolicyNet.evaluate_action(self.State.state[None], torch.tensor(a)) if self.State.state is not None else None
        return a

    def update(self, reward, own_action, opponent_action):
        super(ActorCriticLSTMAgent, self).update(reward)
        self.own_memory[self.play_times - 1] = own_action
        self.opponent_memory[self.play_times - 1] = opponent_action
        self.State.oppo_memory = self.opponent_memory[:self.play_times]

        if self.State.state is not None:
            self.State.next_state = self.State.state_repr(torch.cat([self.opponent_action[1:], torch.as_tensor([opponent_action])]),
                                                          torch.cat([self.own_action[1:], torch.as_tensor([own_action])]))
            self.State.next_state = torch.permute(self.State.next_state.view(-1, self.config.h), (1, 0))  # important
            if 'Worker' not in self.name:
                # push the transition into ReplayBuffer
                self.Memory.push(self.State.state, own_action, self.State.next_state, reward)
                # We need to make our task into the episodic task
                # don't learn without some decent experience
                if not len(self.Memory.memory) < self.config.batch_size:
                    self.Workers.set_batch(self.PolicyNet, self.Memory)
                    self.optimize_model()

    def optimize_model(self):
        """ Train our model """
        transitions = self.Memory.memory
        # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
        state, action, next_state, reward = zip(*transitions)
        # convert to PyTorch and define types
        state = torch.stack(list(state), dim=0).to(device).view(-1, self.config.h, self.input_size)
        next_state = torch.stack(list(next_state), dim=0).to(device).view(-1, self.config.h, self.input_size)
        action = torch.tensor(action, dtype=torch.int64, device=device)  # Need 64 bit to use them as index
        reward = torch.tensor(reward, dtype=torch.float, device=device)[:, None]

        with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
            target = reward + self.config.discount * self.PolicyNet.get_critic(next_state)
        values, log_probs, entropy = self.PolicyNet.evaluate_action(state, action)

        # loss is measured from error between current and V values
        advantages = target - values
        critic_loss = F.smooth_l1_loss(values, target)
        actor_loss = - (log_probs * advantages.detach()).mean()

        total_loss = (CRITIC_COEF * critic_loss) + actor_loss - (ENTROPY_COEF * entropy)

        # backpropagation of loss to Neural Network (PyTorch magic)
        self.Optimizer.zero_grad()
        total_loss.backward()
        for param in self.PolicyNet.parameters():
            param.grad.data.clamp_(-1, 1)  # DQN gradient clipping: Clamps all elements in input into the range [ min, max ].
        self.Optimizer.step()
        self.Memory.clean()
        self.loss.append(total_loss.item())
        # test
        # print(f'==================={self.play_times}===================')
        # print(f'transition: \n{np.hstack((state.numpy(),action.numpy(),next_state.numpy(),reward.numpy()))}')
        # print(f'transition: \nstate: {np.squeeze(state.numpy())}\naction: {np.squeeze(action.numpy())}\nnext_s: {np.squeeze(next_state.numpy())}\nreward: {np.squeeze(reward.numpy())}')
        # print(f'loss: {loss.item()}')

    def show(self):
        print("==================================================")
        print(
            f'Your action: {self.own_memory[self.play_times - 20:self.play_times]}\nOppo action: {self.opponent_memory[self.play_times - 20:self.play_times]}')

class Worker(object):
    def __init__(self, config):
        self.env = Environment(config)
        self.config = config
        self.workers = []
        self.opponents = []
        self.init_workers()

    def init_workers(self):
        self.opponents.append(StrategyAgent('TitForTat',self.config))
        # self.opponents.append(StrategyAgent('revTitForTat',self.config))
        self.opponents.append(StrategyAgent('ALLC',self.config))
        self.opponents.append(StrategyAgent('ALLD',self.config))
        for _ in range(len(self.opponents)):
            self.workers.append(ActorCriticLSTMAgent('Worker', self.config))

    def set_batch(self, PolicyNet, Memory):
        for idx, worker in enumerate(self.workers):
            worker.PolicyNet.load_state_dict(PolicyNet.state_dict())
            worker.PolicyNet.eval()
            agent1, agent2 = worker, self.opponents[idx]
            for i in range(self.config.batch_size):
                a1, a2 = agent1.act(agent2), agent2.act(agent1)
                _, r1, r2 = self.env.step(a1, a2)
                agent1.update(r1, a1, a2)
                agent2.update(r2, a2, a1)
                Memory.push(agent1.State.state, a1, agent1.State.next_state, r1) if agent1.State.state is not None else None
            # print(f'own: {agent2.own_memory}')
            # print(f'oppo: {agent2.opponent_memory}')
