import random
import sys
from tqdm import tqdm
import torch
import torch.nn as nn
from collections import namedtuple, deque

from model import NeuralNetwork
from selection.memory import Memory
from utils import *

TARGET_UPDATE = 10
NUM_HIDDEN = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Agent = namedtuple('Agent', ['agent_1', 'agent_2', 'action_1', 'action_2', 'reward_1', 'reward_2'])
ReplayBuffer = namedtuple('ReplyBuffer', ['state', 'action', 'reward', 'next_state'])

class UpdateMemory(Memory):
    """
    Used for multi-agent games
    """
    def __init__(self, capacity):
        super(UpdateMemory, self).__init__(capacity)

    def push(self, *args):
        self.memory.append(Agent(*args))

class SelectMemory(Memory):
    """
    Replay Buffer
    """
    def __init__(self, capacity):
        super(SelectMemory, self).__init__(capacity)

    def push(self, *args):
        self.memory.append(ReplayBuffer(*args))

def dqn_selection(config, agents, env):
    """
    DQN selection method (benchmark2)
    Parameters
    ----------
    config
    agents
    env

    Returns
    -------

    """
    # construct selection network
    n_agents = len(agents)
    for n in agents:
        agent = agents[n]
        agent.SelectionPolicyNN = NeuralNetwork(n_agents*config.h, n_agents-1, NUM_HIDDEN).to(device)
        agent.SelectionTargetNN = NeuralNetwork(n_agents*config.h, n_agents-1, NUM_HIDDEN).to(device)
        agent.SelectionTargetNN.load_state_dict(agent.SelectionPolicyNN.state_dict())
        agent.SelectMemory = SelectMemory(1000)
        agent.SelectOptimizer = torch.optim.Adam(agent.SelectionPolicyNN.parameters(), lr=config.learning_rate)

    # select using rl based on selection epsilon
    for i in tqdm(range(0, config.n_episodes)):
        society_reward = 0
        update_memory = UpdateMemory(10000)

        # check state
        state = []
        for n in agents:
            agent = agents[n]
            t = agent.play_times
            if t >= config.h:
                state.append(torch.as_tensor(agent.own_memory[t-config.h : t], dtype=torch.float) )
            else: break

        if len(state) != n_agents:
            # select opponent randomly
            for n in agents:
                m = n
                while m == n:
                    m = random.randint(0, n_agents - 1)
                r1, r2 = env.play(agents[n], agents[m], 1)
                society_reward = society_reward + r1 + r2
        else:
            state = torch.stack(state, dim=0).view(n_agents*config.h).to(device)
            # select opponent based on SelectionNN
            for n in agents:
                # select action by epsilon greedy
                sample = random.random()
                m = n
                if sample > agents[n].config.select_epsilon:
                    agents[n].SelectionPolicyNN.eval()
                    a = int(argmax(agents[n].SelectionPolicyNN(state[None])))
                    m = a+1 if a >= n else a
                else:
                    while m == n:
                        m = random.randint(0, n_agents - 1)
                # play
                agent1, agent2 = agents[n], agents[m]
                a1, a2 = agent1.act(agent2), agent2.act(agent1)
                _, r1, r2 = env.step(a1, a2)
                update_memory.push(n, m, a1, a2, r1, r2)
            # update
            for me in update_memory.memory:
                agent1, agent2 = agents[me[0]], agents[me[1]]
                a1, a2, r1, r2 = me[2], me[3], me[4], me[5]
                agent1.update(r1, a1, a2)
                agent2.update(r2, a2, a1)
                society_reward = society_reward + r1 + r2

            next_state = []
            for n in agents:
                agent = agents[n]
                t = agent.play_times
                if t >= config.h:
                    next_state.append(torch.as_tensor(agent.own_memory[t - config.h: t], dtype=torch.float))
            next_state = torch.stack(next_state, dim=0).view(n_agents*config.h).to(device)

            losses = []
            for me in update_memory.memory:
                agent1, agent2 = agents[me[0]], agents[me[1]]
                reward = me[4]
                action = me[1]-1 if me[1]>me[0] else me[1]
                agent1.SelectMemory.push(state, action, reward, next_state)
                loss = optimize_model(agent1, n_agents)
                losses.append(loss) if loss is not None else None
                # Update the target network, copying all weights and biases in DQN
                if agent1.play_times % TARGET_UPDATE == 0:
                    agent1.SelectionTargetNN.load_state_dict(agent1.SelectionPolicyNN.state_dict())

                # epsilon decay
                if agent1.config.select_epsilon > agent1.config.min_epsilon:
                    agent1.config.select_epsilon *= agent1.config.epsilon_decay
            # print(losses) if len(losses) != 0 else None
        env.update(society_reward)
    return agents

def optimize_model(agent, n_agents):
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
    if agent.SelectMemory.__len__() < agent.config.batch_size:
        return None
    # random transition batch is taken from experience replay memory
    transitions = agent.SelectMemory.sample(agent.config.batch_size)
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state = zip(*transitions)
    # convert to PyTorch and define types
    state = torch.stack(list(state), dim=0).to(device).view(agent.config.batch_size, -1)

    next_state = torch.stack(list(next_state), dim=0).to(device).view(agent.config.batch_size, -1)
    action = torch.tensor(action, dtype=torch.int64, device=device)[:, None]  # Need 64 bit to use them as index
    reward = torch.tensor(reward, dtype=torch.float, device=device)[:, None]
    criterion = nn.SmoothL1Loss()
    # compute the q value
    outputs = compute_q_vals(agent.SelectionPolicyNN, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(agent.SelectionTargetNN, reward, next_state, agent.config.discount)

    # loss is measured from error between current and newly expected Q values
    loss = criterion(outputs, target)
    # backpropagation of loss to Neural Network (PyTorch magic)
    agent.SelectOptimizer.zero_grad()
    loss.backward()
    for param in agent.SelectionPolicyNN.parameters():
        param.grad.data.clamp_(-1,1)  # DQN gradient clipping: Clamps all elements in input into the range [ min, max ].
    agent.SelectOptimizer.step()
    return loss.item()



