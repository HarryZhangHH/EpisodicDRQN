import random
import sys
from tqdm import tqdm
import torch
import torch.nn as nn
from collections import namedtuple, deque

from model import LSTMVariant
from selection.memory import UpdateMemory, ReplayBuffer, SettlementMemory
from utils import *

TARGET_UPDATE = 10
HIDDEN_SIZE = 256
BATCH_SIZE = 128
FEATURE_SIZE = 2
NUM_LAYER = 1
SETTLEMENT_PROB = 0.005
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def dqrn_selection_play(config: object, agents: dict, env: object):

    def map_action(a: int):
        # C: 0 -> 1; D: 1 -> -1
        return a*(-2)+1

    n_agents = len(agents)
    for n in agents:
        agent = agents[n]
        agent.SelectionPolicyNet = LSTMVariant(n_agents, HIDDEN_SIZE, NUM_LAYER, FEATURE_SIZE*n_agents, n_agents-1, HIDDEN_SIZE).to(device)
        agent.SelectionTargetNet = LSTMVariant(n_agents, HIDDEN_SIZE, NUM_LAYER, FEATURE_SIZE*n_agents, n_agents-1, HIDDEN_SIZE).to(device)
        agent.SelectionTargetNet.load_state_dict(agent.SelectionPolicyNet.state_dict())
        agent.SelectionMemory = ReplayBuffer(10000)
        agent.SelectionOptimizer = torch.optim.Adam(agent.SelectionPolicyNet.parameters(), lr=config.learning_rate)

    beliefs = np.zeros((n_agents,1))
    # select using rl based on selection epsilon
    for i in tqdm(range(0, config.n_episodes)):
        society_reward = 0
        settlement_memory = SettlementMemory(10000)

        # check the settlement state
        sample = random.random()
        if sample < SETTLEMENT_PROB:
            for me in settlement_memory:
                print(me)
            continue

        # check state: (h_action, features)
        while True:
            h_action = []
            for n in agents:
                agent = agents[n]
                t = agent.play_times
                if t >= config.h:
                    h_action.append(torch.as_tensor(agent.own_memory[t - config.h: t], dtype=torch.float))
                else:
                    break
            if len(h_action) != n_agents:
                # select opponent randomly
                for n in agents:
                    m = n
                    while m == n:
                        m = random.randint(0, n_agents - 1)
                    # play and update
                    agent1, agent2 = agents[n], agents[m]
                    a1, a2, r1, r2 = play(agent1, agent2, env)
                    beliefs[n] = beliefs[n] + config.learning_rate * map_action(int(a1))
                    beliefs[m] = beliefs[m] + config.learning_rate * map_action(int(a2))
            else:
                break

        # process the state
        h_action = torch.stack(h_action, dim=0)
        h_action = h_action.T
        features = torch.from_numpy(beliefs)

        # sample one agent to select and play
        n = np.random.randint(0, n_agents)

        # select opponent based on SelectionNN
        # select action by epsilon greedy
        sample = random.random()
        m = n
        if sample > agents[n].config.select_epsilon:
            agents[n].SelectionPolicyNet.eval()
            s = (h_action[None].to(device), features[None].to(device))
            a = int(argmax(agents[n].SelectionPolicyNet(s)))
            m = a + 1 if a >= n else a
        else:
            while m == n:
                m = random.randint(0, n_agents - 1)

        # play
        agent1, agent2 = agents[n], agents[m]
        a1, a2, r1, r2 = play(agent1, agent2)
        society_reward = society_reward + r1 + r2

        beliefs[n] = beliefs[n] + config.learning_rate * map_action(int(a1))
        beliefs[m] = beliefs[m] + config.learning_rate * map_action(int(a2))

        settlement_memory.push(n, m, a1, a2, r1, r2, agent1.State.state, agent2.State.state)

        # process the state and next_state
        state = (h_action.numpy(), features.numpy())
        h_action = []
        for n in agents:
            agent = agents[n]
            t = agent.play_times
            if t >= config.h:
                h_action.append(torch.as_tensor(agent.own_memory[t - config.h: t], dtype=torch.float))
        h_action = torch.stack(h_action, dim=0)
        h_action = h_action.T
        features = torch.from_numpy(beliefs)
        next_state = (h_action.numpy(), features.numpy())
        action = m - 1 if m > n else m
        reward = r1

        # push trajectories into Selection ReplayBuffer(Memory) and optimize the model
        losses = []
        agent1.SelectionMemory.push(state, action, reward, next_state)
        agent1.SelectionPolicyNet.train()
        loss = __optimize_model(agent1, n_agents)
        losses.append(loss) if loss is not None else None

        # update the target network, copying all weights and biases in DQN
        if agent1.play_times % TARGET_UPDATE == 0:
            agent1.SelectionTargetNet.load_state_dict(agent1.SelectionPolicyNet.state_dict())

        # epsilon decay
        if agent1.config.select_epsilon > agent1.config.min_epsilon:
            agent1.config.select_epsilon *= agent1.config.epsilon_decay



def play(agent1: object, agent2: object, env:object):
    a1, a2 = agent1.act(agent2), agent2.act(agent1)
    _, r1, r2 = env.step(a1, a2)
    agent1.update(r1, a1, a2)
    agent2.update(r2, a2, a1)
    return a1, a2, r1, r2

def __optimize_model(agent: object, n_agents: int):
    """ Train and optimize our model """
    # don't learn without some decent experience
    if agent.SelectionMemory.__len__() < BATCH_SIZE:
        return None
    # random transition batch is taken from experience replay memory
    transitions = agent.SelectionMemory.sample(BATCH_SIZE)
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state = zip(*transitions)
    # convert to PyTorch and define types
    h_action = torch.from_numpy(np.vstack(np.array(state, dtype=object)[:,0]).astype(np.float)).view(BATCH_SIZE, agent.config.h, n_agents).to(device)
    features = torch.from_numpy(np.vstack(np.array(state, dtype=object)[:,1]).astype(np.float)).view(BATCH_SIZE, FEATURE_SIZE*n_agents).to(device)
    state = (h_action, features)
    h_action = torch.from_numpy(np.vstack(np.array(next_state, dtype=object)[:,0]).astype(np.float)).view(BATCH_SIZE, agent.config.h, n_agents).to(device)
    features = torch.from_numpy(np.vstack(np.array(next_state, dtype=object)[:,1]).astype(np.float)).view(BATCH_SIZE, FEATURE_SIZE*n_agents).to(device)
    next_state = (h_action, features)

    action = torch.tensor(action, dtype=torch.int64, device=device)[:, None]  # Need 64 bit to use them as index
    reward = torch.tensor(reward, dtype=torch.float, device=device)[:, None]
    criterion = nn.SmoothL1Loss()  # Compute Huber loss
    # compute the q value
    outputs = compute_q_vals(agent.SelectionPolicyNet, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(agent.SelectionTargetNet, reward, next_state, agent.config.discount)

    # loss is measured from error between current and newly expected Q values
    loss = criterion(outputs, target)
    # backpropagation of loss to Neural Network (PyTorch magic)
    agent.SelectionOptimizer.zero_grad()
    loss.backward()
    for param in agent.SelectionPolicyNet.parameters():
        param.grad.data.clamp_(-1,1)  # DQN gradient clipping: Clamps all elements in input into the range [ min, max ].
    agent.SelectionOptimizer.step()
    return loss.item()










