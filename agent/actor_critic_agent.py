import torch.nn.functional as F
from component.env import Environment
from agent.abstract_agent import AbstractAgent
from model import A2CNetwork
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HIDDEN_SIZE = 128
BUFFER_SIZE = 1000
TARGET_UPDATE = 10
ENTROPY_COEF = 0.01
CRITIC_COEF = 0.5
WORKER_NUM = 16

class ActorCriticAgent(AbstractAgent):
    # h is every agents' most recent h actions are visiable to others which is composed to state
    def __init__(self, name: str, config: object):
        """

        Parameters
        ----------
        config
        name = A2C
        """
        super(ActorCriticAgent, self).__init__(config)
        self.name = name
        self.own_memory = torch.zeros((config.n_episodes*1000, ))
        self.opponent_memory = torch.zeros((config.n_episodes*1000, ))
        self.State = self.StateRepr(method=config.state_repr)
        self.build()
        self.loss = []

    def build(self):
        """State, Policy, Memory, Q are objects"""
        input_size = self.config.h if self.config.state_repr == 'uni' else self.config.h * 2 if self.config.state_repr == 'bi' else 1
        self.PolicyNet = A2CNetwork(input_size, self.config.n_actions, HIDDEN_SIZE).to(device)  # an object

        if 'Worker' not in self.name:
            print(self.PolicyNet.eval())
            self.Memory = self.ReplayBuffer(BUFFER_SIZE)  # an object
            self.Optimizer = torch.optim.Adam(self.PolicyNet.parameters(), lr=self.config.learning_rate)
            self.Workers = Worker(self.config)

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
        # selection action based on epsilon greedy policy
        a = self.PolicyNet.act(self.State.state) if self.State.state is not None else random.randint(0, self.config.n_actions-1)
        # self.PolicyNet.evaluate_action(self.State.state[None], torch.tensor(a)) if self.State.state is not None else None
        return a

    def update(self, reward: float, own_action: int, opponent_action: int):
        super(ActorCriticAgent, self).update(reward)
        self.own_memory[self.play_times - 1] = own_action
        self.opponent_memory[self.play_times - 1] = opponent_action
        # self.State.oppo_memory = self.opponent_memory[:self.play_times]

    def optimize(self, action: int, reward: float, oppo_agent: object, state=None):
        super(ActorCriticAgent, self).optimize(action, reward, oppo_agent)
        if self.State.state is None:
            return None

        opponent_h_actions = torch.as_tensor(
            oppo_agent.own_memory[oppo_agent.play_times - self.config.h: oppo_agent.play_times])
        own_h_actions = torch.as_tensor(
            self.own_memory[self.play_times - self.config.h: self.play_times])

        self.State.next_state = self.State.state_repr(opponent_h_actions, own_h_actions)
        if 'Worker' not in self.name:
            # push the transition into ReplayBuffer
            self.Memory.push(self.State.state, action, self.State.next_state, reward)
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
        state = torch.stack(list(state), dim=0).to(device)
        action = torch.tensor(action, dtype=torch.int64, device=device)  # Need 64 bit to use them as index
        next_state = torch.stack(list(next_state), dim=0).to(device)
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
    def __init__(self, config: object):
        self.env = Environment(config)
        self.config = config
        self.workers = []
        self.opponents = []
        self.init_workers()

    def init_workers(self):
        self.opponents.append(StrategyAgent('TitForTat',self.config))
        self.opponents.append(StrategyAgent('Pavlov',self.config))
        for _ in range(len(self.opponents)):
            self.workers.append(ActorCriticAgent('Worker', self.config))

    def set_batch(self, PolicyNet: object, Memory: object):
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
