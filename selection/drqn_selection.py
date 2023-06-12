import torch.nn as nn

from model import LSTMVariant, DQN, DDQN
from component.env import Environment, StochasticGameEnvironment
from component.memory import ReplayBuffer, Memory, RecordMemory
from selection.maxmin_drqn_selection import update_convergence_episode
from utils import *

TARGET_UPDATE = 10
HIDDEN_SIZE = 128
BATCH_SIZE = 64
FEATURE_SIZE = 1
BUFFER_SIZE = 1000
NUM_LAYER = 1
UPDATE_TIMES = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def drqn_selection(config: object, agents: dict, k:int = 1000, episodic_flag: bool = True, sg_flag: bool = False, select_method='DQN', play_method='DQN'):
    """
    DQN-based selection method - using LSTM or ensemble LSTM (LSTM-VARIANT)

    Parameters
    ----------
    config: object
    agents: dict[object]
        dictionary of n unupdated agents
    env: object

    Returns
    -------
    agents: dict[object]
        dictionary of n updated agents
    """
    # Initialize convergence threshold
    count = 0
    test_state = generate_state(agents[0], config.h, config.n_actions, k)
    test_state = torch.stack(test_state, dim=0).view(k, config.h, -1).to(device)
    thresh_strategy = k * config.min_epsilon + 5
    thresh_network = k / 100
    thresh_reward = 1
    thresh = (thresh_strategy, thresh_reward, thresh_network)

    # initialize running log
    select_dict = {}
    selected_dict = {}
    test_q_dict = {}
    for n in agents:
        select_dict[n] = 0
        selected_dict[n] = 0
        test_q_dict[n] = {}

    convergent_episode_dict = {'strategy': {}, 'reward': {}, 'network': {}}
    authority = CentralAuthorityNormal(config, agents, k, episodic_flag=episodic_flag, sg_flag=sg_flag,
                                 select_method=select_method, play_method=play_method)

    while True:
        last_reward = {}
        for n in agents:
            last_reward[n] = agents[n].running_score

        select_dict, selected_dict = authority.run(select_dict, selected_dict)
        agents = authority.agents

        strategy_convergent_episode, reward_convergent_episode, network_convergent_episode, test_q_dict = check_convergence(
            agents, test_state, thresh, k, last_reward, test_q_dict, count)
        count += 1

        update_convergence_episode(convergent_episode_dict, 'strategy', strategy_convergent_episode)
        update_convergence_episode(convergent_episode_dict, 'reward', reward_convergent_episode)
        update_convergence_episode(convergent_episode_dict, 'network', network_convergent_episode)

        print(convergent_episode_dict)

        if count * k >= config.n_episodes:
            print('Not Convergent')
            for n in agents:
                print(f'Agent {n} updating times: {agents[n].updating_times}')
            break

        converge = []
        for key, _ in convergent_episode_dict.items():
            converge.append(len(convergent_episode_dict[key]) == len(agents))
        if sum(converge) >= 2:
            for n in agents:
                print(f'Agent {n} updating times: {agents[n].updating_times}')
            break

    return agents, select_dict, selected_dict, authority.beliefs, count, convergent_episode_dict, authority.env


class CentralAuthorityNormal():
    def __init__(self, config: object, agents: dict[int, object], k: int, episodic_flag: bool = True, sg_flag: bool = False, settlement_prob: float = 0.005, select_epsilon_decay: float = 0.999, update_times: int = 20, select_method: str = 'DDQN', play_method: str = 'DDQN'):
        self.config = config
        self.n_agents = len(agents)
        self.k = k
        self.episodic_flag = episodic_flag
        self.update_times = update_times if episodic_flag else 1
        self.sg_flag = sg_flag
        self.settlement_prob = settlement_prob
        self.select_h = self.config.select_h
        self.select_epsilon_decay = select_epsilon_decay
        self.selection_learning_rate = config.learning_rate
        self.env = Environment(config) if not self.sg_flag else StochasticGameEnvironment(config, thresh=1)
        self.agents = self.initialize_agent_configuration(agents)
        self.beliefs = np.zeros((len(agents),1))
        self.record_memory = RecordMemory(100000)
        self.state = 1
        self.selection_loss_dict = {}
        self.select_model = DDQN() if select_method=='DDQN' else DQN()
        self.model = DDQN() if play_method=='DDQN' else DQN()

    def initialize_agent_configuration(self, agents: dict[int, object]):
        for n in agents:
            agent = agents[n]
            agent.selection_policy_net = LSTMVariant(self.n_agents, HIDDEN_SIZE, NUM_LAYER, FEATURE_SIZE*self.n_agents,
                                                   self.n_agents-1, HIDDEN_SIZE).to(device)
            agent.selection_target_net = LSTMVariant(self.n_agents, HIDDEN_SIZE, NUM_LAYER, FEATURE_SIZE*self.n_agents,
                                                   self.n_agents-1,HIDDEN_SIZE).to(device)
            agent.selection_target_net.load_state_dict(agent.selection_policy_net.state_dict())
            agent.selection_memory = ReplayBuffer(BUFFER_SIZE)
            agent.selection_optimizer = torch.optim.Adam(agent.selection_policy_net.parameters(),
                                                        lr=self.selection_learning_rate)
            agent.selection_target_net.eval()
            agent.select_epsilon = agent.config.select_epsilon
            agent.updating_times = 0
        return agents

    @staticmethod
    def map_action(a: int):
        # C: 0 -> 1; D: 1 -> -1
        return a * (-2) + 1

    def check_state(self):
        # check the settlement state
        sample = random.random()
        self.state = 1
        if self.sg_flag:
            self.state = self.env.check_state(self.agents)
        if sample < self.settlement_prob:
            self.state = -1

    def update_belief(self, n, m, a1, a2):
        self.beliefs[n] += self.selection_learning_rate * self.map_action(int(a1))
        self.beliefs[m] += self.selection_learning_rate * self.map_action(int(a2))

    def play(self, agent1: object, agent2: object, env: object):
        a1, a2 = agent1.act(agent2), agent2.act(agent1)

        # check the SG state
        agents = {}
        agents[0], agents[1] = agent1, agent2
        env.update_state(agents)

        _, r1, r2 = env.step(a1, a2)
        agent1.update(r1, a1, a2)
        agent2.update(r2, a2, a1)
        return a1, a2, r1, r2

    def select_opponent(self, n, s):
        # select action by epsilon greedy
        sample = random.random()
        m = n
        if sample > self.agents[n].config.select_epsilon:
            self.agents[n].selection_policy_net.eval()
            a = int(argmax(self.agents[n].selection_policy_net(s)))  # select opponent based on selection_policy_net
            m = a + 1 if a >= n else a
        else:
            while m == n:
                m = random.randint(0, self.n_agents - 1)
        return m

    def run(self, select_dict: dict[int, int], selected_dict: dict[int, int]):
        for n in self.agents:
            self.selection_loss_dict[n] = []
        # select using rl based on selection epsilon
        for i in range(0, self.k):
            # check the settlement state
            self.check_state()
            if self.state==-1 and self.episodic_flag:
                # print(f'======= Episode: {i} SETTLEMENT =======')
                self.__optimize_play_model()
                # update the target network, copying all weights and biases in DRQN
                for n in self.agents:
                    agent = self.agents[n]
                    agent.target_net.load_state_dict(agent.policy_net.state_dict())
                self.beliefs = np.clip(self.beliefs, -1, 1)  # clip belief into [-1,1]

                # clean the buffer
                for n in self.agents:
                    self.agents[n].selection_memory.clean()
                for n in self.agents:
                    self.agents[n].memory.clean()
                self.env.reset_state()

            # check selection state: (h_action, features)
            while True:
                h_action = []
                for n in self.agents:
                    agent = self.agents[n]
                    t = agent.play_times
                    if t >= self.select_h:
                        h_action.append(torch.as_tensor(agent.own_memory[t-self.select_h: t], dtype=torch.float))
                    else:
                        break
                if len(h_action) != self.n_agents:
                    # select opponent randomly
                    for n in self.agents:
                        m = n
                        while m == n:
                            m = random.randint(0, self.n_agents - 1)
                        # play and update
                        agent1, agent2 = self.agents[n], self.agents[m]
                        a1, a2, r1, r2 = self.play(agent1, agent2, self.env)
                        self.update_belief(n, m, a1, a2)
                        self.record_memory.push(n, m, None, a1, a2, r1, r2)
                else:
                    break

            # process the state
            h_action = torch.stack(h_action, dim=0)
            h_action = h_action.T
            features = torch.from_numpy(self.beliefs)
            s = (h_action[None].to(device), features[None].to(device))

            # sample one agent to select and play
            n = np.random.randint(0, self.n_agents)
            m = self.select_opponent(n, s)
            select_dict[n] += 1
            selected_dict[m] += 1
            # play
            agent1, agent2 = self.agents[n], self.agents[m]
            a1, a2, r1, r2 = self.play(agent1, agent2, self.env)
            self.env.update(r1 + r2)
            agent1.optimize(a1, r1, agent2, flag=not self.episodic_flag)
            agent2.optimize(a2, r2, agent1, flag=not self.episodic_flag)

            # update beliefs
            self.update_belief(n, m, a1, a2)
            self.env.update(r1+r2)

            if len(self.agents) > 2:
                # process the state, action, reward and next_state
                state = (h_action.numpy(), features.numpy())
                h_action = []
                for idx in self.agents:
                    agent = self.agents[idx]
                    t = agent.play_times
                    if t >= self.select_h:
                        h_action.append(torch.as_tensor(agent.own_memory[t - self.select_h: t], dtype=torch.float))
                h_action = torch.stack(h_action, dim=0)
                h_action = h_action.T
                features = torch.from_numpy(self.beliefs)
                next_state = (h_action.numpy(), features.numpy())
                action = m - 1 if m > n else m
                reward = r1

                # push trajectories into Selection ReplayBuffer(Memory) and optimize the model
                agent1.selection_memory.push(state, action, reward, next_state)
                self.record_memory.push(n, m, state, a1, a2, r1, r2)
                agent1.selection_policy_net.train()
                loss = self.__optimize_selection_model(agent1)
                self.selection_loss_dict[n].append(loss) if loss is not None else None

                # update the target network, copying all weights and biases in DRQN
                if agent1.play_times % TARGET_UPDATE == 0:
                    agent1.selection_target_net.load_state_dict(agent1.selection_policy_net.state_dict())

                # selection policy epsilon decay
                if agent1.select_epsilon > self.config.min_epsilon:
                    agent1.select_epsilon *= self.select_epsilon_decay
                if agent1.select_epsilon <= self.config.min_epsilon:
                    agent1.select_epsilon = self.config.min_epsilon

        return select_dict, selected_dict

    def __optimize_selection_model(self, agent: object):
        """ Train and optimize our selection model """
        # don't learn without some decent experience
        if agent.selection_memory.__len__() < BATCH_SIZE:
            return None
        # random transition batch is taken from experience replay memory
        transitions = agent.selection_memory.sample(BATCH_SIZE)
        # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
        state, action, reward, next_state = zip(*transitions)
        # convert to PyTorch and define types
        h_action = torch.from_numpy(np.vstack(np.array(state, dtype=object)[:,0]).astype(np.float)).view(BATCH_SIZE, self.select_h, self.n_agents).to(device)
        features = torch.from_numpy(np.vstack(np.array(state, dtype=object)[:,1]).astype(np.float)).view(BATCH_SIZE, max(1,FEATURE_SIZE)*self.n_agents).to(device)
        state = (h_action, features)
        h_action = torch.from_numpy(np.vstack(np.array(next_state, dtype=object)[:,0]).astype(np.float)).view(BATCH_SIZE, self.select_h, self.n_agents).to(device)
        features = torch.from_numpy(np.vstack(np.array(next_state, dtype=object)[:,1]).astype(np.float)).view(BATCH_SIZE, max(1,FEATURE_SIZE)*self.n_agents).to(device)
        next_state = (h_action, features)
        action = torch.tensor(action, dtype=torch.int64, device=device)[:, None]  # Need 64 bit to use them as index
        reward = torch.tensor(reward, dtype=torch.float, device=device)[:, None]

        criterion = nn.SmoothL1Loss()  # Compute Huber loss
        batch = Memory.Transition(state, action, next_state, reward)
        loss = self.select_model.optimize(agent.selection_policy_net, agent.selection_target_net, agent.selection_optimizer, batch, agent.config.discount, criterion)
        return loss.item()

    def __optimize_play_model(self):
        """ Train and optimize all agents' play model """
        # Train and optimize play model
        for n in self.agents:
            agent = self.agents[n]
            if len(agent.memory) < agent.config.batch_size:
                continue
            else:
                for _ in range(self.update_times):
                    # random transition batch is taken from experience replay memory
                    transitions = agent.memory.sample(agent.config.batch_size)
                    batch = agent.get_batch(transitions)
                    agent.model.train(agent, batch)
                    agent.policy.update_epsilon(agent.config)
                    agent.updating_times += 1

def check_convergence(agents: dict[int, object], test_state: Type.TensorType, thresh: tuple, k: int, last_reward: dict[int, float], test_q_dict: dict, count: int):
    strategy_convergent_episode = {}
    reward_convergent_episode = {}
    network_convergent_episode = {}
    thresh_strategy, thresh_reward, thresh_network = thresh[0], thresh[1], thresh[2]

    for n in agents:
        # Strategy convergence
        strategy_convergent = agents[n].determine_convergence(thresh_strategy, k)
        if strategy_convergent:
            strategy_convergent_episode[n] = agents[n].play_times

        # reward convergence
        if np.abs(agents[n].running_score-last_reward[n]) <= thresh_reward:
            reward_convergent_episode[n] = agents[n].play_times

        with torch.no_grad():
            test_q_dict[n][count] = agents[n].policy_net(test_state).to('cpu').detach().numpy()

        diff = np.sum(np.diff(test_q_dict[n][count])) - np.sum(np.diff(test_q_dict[n][count-1])) if count>1 else np.inf

        # network convergence
        if np.abs(diff) < thresh_network:
            network_convergent_episode[n] = agents[n].play_times
    return strategy_convergent_episode, reward_convergent_episode, network_convergent_episode, test_q_dict