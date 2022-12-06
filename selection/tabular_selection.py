import numpy as np
import torch
import random
from utils import label_encode, argmax, iterate_combination
from tqdm import tqdm
from collections import namedtuple, deque
from selection.memory import Memory

H = 2
Agent = namedtuple('Agent', ['state', 'action', 'agent_1', 'agent_2', 'action_1', 'action_2', 'reward_1', 'reward_2'])

class SelectMemory(Memory):
    """
    Used for multi-agent games
    """
    def __init__(self, capacity):
        super(SelectMemory, self).__init__(capacity)

    def push(self, *args):
        self.memory.append(Agent(*args))

def tabular_selection(config: object, agents: dict[object], env: object):
    """
    Tabular selection method (benchmark)
    Args
    -------
    config: object
    agents: dict
        the agents dictionary
    env: object

    Returns
    -------
    agents: dict of objects
    """
    n_agents = len(agents)

    # initialize Q table
    h = min(config.h, H)
    num = 2**h
    state_list = iterate_combination(num)
    Q_table = torch.full((len(state_list), num), float('-inf'))
    for idx, val in enumerate(state_list):
        Q_table[idx, list(val)] = 0

    # select opponent randomly at first
    for i in range(config.h):
        society_reward = 0
        for n in range(n_agents):
            m = n
            while m == n:
                m = random.randint(0, n_agents-1)
            r1, r2 = env.play(agents[n], agents[m], 1)
            society_reward = society_reward + r1 + r2
            # print(f"Agent {n}, {agents[n].name}, score: {agents[n].running_score}, play times: {agents[n].play_times}", end=' ')
            # print("  v.s.  ", end='')
            # print(f"Agent {m}, {agents[m].name}, score: {agents[m].running_score}, play times: {agents[m].play_times}")
        env.update(society_reward)

    # get history action from agents' memory
    action_hist = torch.zeros((n_agents, h))
    for n in range(n_agents):
        t = agents[n].play_times
        action_hist[n, :] = torch.as_tensor(agents[n].own_memory[t - h:t])
    action_hist = label_encode(action_hist.T)

    # select using rl based on selection epsilon
    for i in tqdm(range(config.h, config.n_episodes)):
        society_reward = 0
        memory = SelectMemory(10000)

        # select opponent and play
        for n in range(n_agents):
            # select the agent
            state_encode = action_hist[torch.arange(0, action_hist.shape[0]) != n, ...]
            state_encode = tuple(torch.unique(state_encode, sorted=True).tolist())
            state = state_list.index(state_encode)

            # select action by epsilon greedy
            sample = random.random()
            m = n
            if sample > config.select_epsilon:
                action_encode = argmax(Q_table[state])
                while m == n:
                    idx_list = [i for i, x in enumerate(action_hist.tolist()) if x == action_encode]
                    m = random.choice(idx_list)
            else:
                while m == n:
                    m = random.randint(0, n_agents-1)
                action_encode = action_hist[m]

            # play
            agent1, agent2 = agents[n], agents[m]
            a1, a2 = agent1.act(agent2), agent2.act(agent1)
            episode, r1, r2 = env.step(a1, a2)
            # store the data into the select buffer and update all the Q_table after all agents play
            # Agent = namedtuple('Agent', ['state', 'action', 'agent_1', 'agent_2', 'action_1', 'action_2', 'reward_1', 'reward_2'])
            memory.push(state, action_encode, n, m, a1, a2, r1, r2)

        # update the Q table
        for me in memory.memory:
            agent1, agent2 = agents[me[2]], agents[me[3]]
            a1, a2, r1, r2 = me[4], me[5], me[6], me[7]
            agent1.update(r1, a1, a2)
            agent2.update(r2, a2, a1)
            society_reward = society_reward + r1 + r2

        # get history action from agents' memory
        action_hist = torch.zeros((n_agents, h))
        for n in range(n_agents):
            t = agents[n].play_times
            action_hist[n, :] = torch.as_tensor(agents[n].own_memory[t - h:t])
        action_hist = label_encode(action_hist.T)

        for me in memory.memory:
            state, action, reward, agent_idx = me[0], me[1], me[6], me[3]
            next_state = action_hist[torch.arange(0, action_hist.shape[0]) != agent_idx, ...]
            Q_table[state, action] = (1 - config.alpha) * Q_table[state, action] \
                                     + config.alpha * (reward + config.discount * (torch.max(Q_table[next_state])))

        env.update(society_reward)

        # epsilon decay
        if config.select_epsilon > config.min_epsilon:
            config.select_epsilon *= config.epsilon_decay

    print('Q table: \n{}'.format(Q_table))
    return agents
