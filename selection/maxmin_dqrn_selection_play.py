import random
import sys
from tqdm import tqdm
import torch
import copy
import torch.nn as nn
from collections import namedtuple, deque

from model import LSTMVariant
from selection.memory import UpdateMemory, ReplayBuffer, SettlementMemory
from utils import *

TARGET_UPDATE = 10
HIDDEN_SIZE = 128
BATCH_SIZE = 64
FEATURE_SIZE = 1
BUFFER_SIZE = 1000
NUM_LAYER = 1
SETTLEMENT_PROB = 0.005
UPDATE_TIMES = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def maxmin_dqrn_selection_play(config: object, agents: dict, env: object):

    def map_action(a: int):
        # C: 0 -> 1; D: 1 -> -1
        return a*(-2)+1

    n_agents = len(agents)
    select_dict = {}
    selected_dict = {}
    for n in range(n_agents):
        select_dict[n] = 0
        selected_dict[n] = 0


    initialize_agent_configuration(agents)

    beliefs = np.zeros((n_agents,1))
    settlement_memory = SettlementMemory(10000)

    # select using rl based on selection epsilon
    for i in tqdm(range(0, config.n_episodes)):
        society_reward = 0

        # check the settlement state
        sample = random.random()
        if sample < SETTLEMENT_PROB:
            # print(f'======= Episode: {i} SETTLEMENT =======')
            __optimize_play_model(agents, settlement_memory)
            for n in agents:
                agents[n].SelectionMemory.clean()
            settlement_memory.clean()
            beliefs = np.clip(beliefs, -1, 1)   # clip belief into [-1,1]
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

        # select action by epsilon greedy
        sample = random.random()
        m = n
        if sample > agents[n].config.select_epsilon:
            agents[n].SelectionPolicyNet.eval()
            s = (h_action[None].to(device), features[None].to(device))
            a = int(argmax(agents[n].SelectionPolicyNet(s)))  # select opponent based on SelectionPolicyNet
            m = a + 1 if a >= n else a
        else:
            while m == n:
                m = random.randint(0, n_agents - 1)

        select_dict[n] += 1
        selected_dict[m] += 1

        # play
        agent1, agent2 = agents[n], agents[m]
        a1, a2, r1, r2 = play(agent1, agent2, env)
        society_reward = society_reward + r1 + r2

        # update beliefs
        beliefs[n] = beliefs[n] + config.learning_rate * map_action(int(a1))
        beliefs[m] = beliefs[m] + config.learning_rate * map_action(int(a2))

        agent1.get_next_state(agent2)
        agent2.get_next_state(agent1)

        settlement_memory.push(n, m, a1, a2, r1, r2, agent1.State.state, agent2.State.state, agent1.State.next_state, agent2.State.next_state)

        # process the state, action, reward and next_state
        state = (h_action.numpy(), features.numpy())
        h_action = []
        for idx in agents:
            agent = agents[idx]
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
        agent1.SelectionMemoryLog.push(state, action, reward, next_state)
        agent1.SelectionPolicyNet.train()
        loss = __optimize_selection_model(agent1, n_agents)
        losses.append(loss) if loss is not None else None

        # update the target network, copying all weights and biases in DQN
        if agent1.play_times % TARGET_UPDATE == 0:
            agent1.SelectionTargetNet.load_state_dict(agent1.SelectionPolicyNet.state_dict())

        # epsilon decay
        if agent1.config.select_epsilon > agent1.config.min_epsilon:
            agent1.config.select_epsilon *= agent1.config.epsilon_decay
        if agent1.config.select_epsilon <= agent1.config.min_epsilon:
            agent1.config.select_epsilon = agent1.config.min_epsilon

    return agents, select_dict, selected_dict, beliefs, losses

def initialize_agent_configuration(agents:dict):
    n_agents = len(agents)
    for n in agents:
        agent = agents[n]
        agent.play_memory = {}
        agent.play_policy_net = {}
        agent.play_target_net = {}
        agent.play_optimizer = {}
        agent.play_loss = {}
        agent.SelectionPolicyNet = LSTMVariant(n_agents, HIDDEN_SIZE, NUM_LAYER, FEATURE_SIZE * n_agents, n_agents - 1,
                                               HIDDEN_SIZE).to(device)
        agent.SelectionTargetNet = LSTMVariant(n_agents, HIDDEN_SIZE, NUM_LAYER, FEATURE_SIZE * n_agents, n_agents - 1,
                                               HIDDEN_SIZE).to(device)
        agent.SelectionTargetNet.load_state_dict(agent.SelectionPolicyNet.state_dict())
        agent.SelectionMemory = ReplayBuffer(10000)
        agent.SelectionMemoryLog = ReplayBuffer(10000)
        agent.SelectionOptimizer = torch.optim.Adam(agent.SelectionPolicyNet.parameters(), lr=agent.config.learning_rate)
        agent.SelectionTargetNet.eval()

        # for name, param in agent.SelectionPolicyNet.named_parameters():
        #     print(name, param.data)

def play(agent1: object, agent2: object, env:object):
    a1, a2 = agent1.act(agent2), agent2.act(agent1)
    if agent1.State.state is not None and agent2.State.state is not None:
        a1 = get_action_selection_q_values(agent1, agent1.State.state)
        a2 = get_action_selection_q_values(agent2, agent2.State.state)

    _, r1, r2 = env.step(a1, a2)
    agent1.update(r1, a1, a2)
    agent2.update(r2, a2, a1)
    return a1, a2, r1, r2

def __optimize_selection_model(agent: object, n_agents: int):
    """ Train and optimize our selection model """
    # don't learn without some decent experience
    if agent.SelectionMemory.__len__() < BATCH_SIZE:
        return None
    # random transition batch is taken from experience replay memory
    transitions = agent.SelectionMemory.sample(BATCH_SIZE)
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state = zip(*transitions)
    # convert to PyTorch and define types
    h_action = torch.from_numpy(np.vstack(np.array(state, dtype=object)[:,0]).astype(np.float)).view(BATCH_SIZE, agent.config.h, n_agents).to(device)
    features = torch.from_numpy(np.vstack(np.array(state, dtype=object)[:,1]).astype(np.float)).view(BATCH_SIZE, max(1,FEATURE_SIZE)*n_agents).to(device)
    state = (h_action, features)
    h_action = torch.from_numpy(np.vstack(np.array(next_state, dtype=object)[:,0]).astype(np.float)).view(BATCH_SIZE, agent.config.h, n_agents).to(device)
    features = torch.from_numpy(np.vstack(np.array(next_state, dtype=object)[:,1]).astype(np.float)).view(BATCH_SIZE, max(1,FEATURE_SIZE)*n_agents).to(device)
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

def __optimize_play_model(agents:dict, settlement_memory:object):
    """ Train and optimize all agents' play model """

    # Extract the data from settlement memory and store them in the play memory by the opponent index
    for me in settlement_memory.memory:
        n, m = me[0], me[1]
        agent1, agent2 = agents[n], agents[m]
        a1, a2, r1, r2, s1, s2, next_s1, next_s2 = me[2], me[3], me[4], me[5], me[6], me[7], me[8], me[9]

        # check fix strategy
        if s1 is not None and next_s1 is not None:
            if m not in agent1.play_memory.keys():
                agent1.play_loss[m] = []
                agent1.play_memory[m] = agent1.ReplayBuffer(BUFFER_SIZE)
                agent1.play_policy_net[m] = copy.deepcopy(agent1.PolicyNet)
                agent1.play_target_net[m] = copy.deepcopy(agent1.TargetNet)
                agent1.play_target_net[m].load_state_dict(agent1.play_policy_net[m].state_dict())
                agent1.play_target_net[m].eval()
                agent1.play_optimizer[m] = torch.optim.Adam(agent1.play_policy_net[m].parameters(),
                                                            lr=agent1.config.learning_rate)
            agent1.play_memory[m].push(s1, a1, next_s1, r1)

        if s2 is not None and next_s2 is not None:
            if n not in agent2.play_memory.keys():
                agent2.play_loss[n] = []
                agent2.play_memory[n] = agent2.ReplayBuffer(BUFFER_SIZE)
                agent2.play_policy_net[n] = copy.deepcopy(agent2.PolicyNet)
                agent2.play_target_net[n] = copy.deepcopy(agent2.TargetNet)
                agent2.play_target_net[n].load_state_dict(agent2.play_policy_net[n].state_dict())
                agent2.play_target_net[n].eval()
                agent2.play_optimizer[n] = torch.optim.Adam(agent2.play_policy_net[n].parameters(),
                                                            lr=agent2.config.learning_rate)
            agent2.play_memory[n].push(s2, a2, next_s2, r2)

    if settlement_memory.__len__() != 0:
        del agent1, agent2, a1, a2, r1, r2, s1, s2, next_s1, next_s2

    # Train and optimize play model
    for n in agents:
        agent = agents[n]
        for m in agent.play_memory:
            if len(agent.play_memory[m].memory) < agent.config.batch_size:
                continue
            else:
                for _ in range(UPDATE_TIMES):
                    # random transition batch is taken from experience replay memory
                    transitions = agent.play_memory[m].sample(agent.config.batch_size)
                    criterion, state, action, reward, next_state = agent.get_batch(transitions)
                    outputs = compute_q_vals(agent.play_policy_net[m], state, action)
                    target = compute_q_target(agent, reward, next_state)
                    loss = criterion(outputs, target)

                    # backpropagation of loss to Neural Network (PyTorch magic)
                    agent.play_optimizer[m].zero_grad()
                    loss.backward()
                    for param in agent.play_policy_net[m].parameters():
                        param.grad.data.clamp_(-1,1)  # DQN gradient clipping: Clamps all elements in input into the range [ min, max ].
                    agent.play_optimizer[m].step()
                    agent.play_loss[m].append(loss.item())

                    # update play_epsilon
                    agent.Policy.update_epsilon(agent.config)

            # update the target network, copying all weights and biases in DRQN
            agent.play_target_net[m].load_state_dict(agent.play_policy_net[m].state_dict())

    # clean the play_buffer
    for n in agents:
        for m in agent.play_memory:
            agent.play_memory[m].clean()

def compute_q_target(agent: object, reward: Type.TensorType , next_state: Type.TensorType):
    with torch.no_grad():
        q_min = agent.play_target_net[list(agent.play_target_net.keys())[0]](next_state).clone()
        for m in agent.play_target_net:
            q = agent.play_target_net[m](next_state)
            q_min = torch.min(q_min, q)
        q_next = q_min.max(1)[0]
        q_target = reward + agent.config.discount * q_next[:,None]
    return q_target

def get_action_selection_q_values(agent, state):
    if state is None:
        return random.randint(0, agent.config.n_actions - 1)

    if not list(agent.play_policy_net.keys()):
        return random.randint(0, agent.config.n_actions - 1)

    if not isinstance(state, tuple):
        state = state[None]
    else:
        state = (state[0][None], state[1][None])  # used by LSTMVariant network

    q_min = agent.play_policy_net[list(agent.play_policy_net.keys())[0]](state).clone()
    for m in agent.play_policy_net:
        q = agent.play_policy_net[m](state)
        q_min = torch.min(q_min, q)
    q_min = q_min.cpu().detach().numpy().flatten()
    action = agent.Policy.sample_action(state, q_min)
    return action










