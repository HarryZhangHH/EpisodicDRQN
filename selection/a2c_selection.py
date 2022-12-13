import sys
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque
import numpy as np

from model import FeatureNet, CriticNet, ActorNet
from selection.memory import UpdateMemory, ReplayBuffer
from utils import *

TARGET_UPDATE = 10
HIDDEN_SIZE = 256
BATCH_SIZE = 128
FEATURE_SIZE = 4
NUM_LAYER = 1
ALPHA = 0.01
ENTROPY_COEF = 0.01
CRITIC_COEF = 0.5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def a2c_selection(config: object, agents: dict, env: object):
    """
    SAC selection method using LSTM-Variant in sequential way

    Parameters
    ----------
    config: object
    agents: dict[object]
        dictionary of n unupdated agents
    env: object

    Returns
    -------
    agents: dict[object]
        dictionary of n unupdated agents
    """

    n_agents = len(agents)
    max_reward = config.temptation / (1 - config.discount)  # sum of geometric progression
    max_play_times = config.n_episodes*5/n_agents

    SelectionFeatureNet = FeatureNet(n_agents, HIDDEN_SIZE, NUM_LAYER, FEATURE_SIZE * n_agents).to(device)
    SelectionCriticNet = CriticNet(HIDDEN_SIZE*2, HIDDEN_SIZE).to(device)
    # SelectionMemory = ReplayBuffer(10000)
    # FeatureOptimizer = torch.optim.Adam([{'params': SelectionCriticNet.parameters()}], lr=config.learning_rate)
    CriticOptimizer = torch.optim.Adam([
                {'params': SelectionFeatureNet.parameters()},
                {'params': SelectionCriticNet.parameters()},
            ], lr=config.learning_rate)

    for n in agents:
        agent = agents[n]
        agent.SelectionActorNet = ActorNet(HIDDEN_SIZE*2, HIDDEN_SIZE, n_agents-1).to(device)
        agent.SelectionOptimizer = torch.optim.Adam([
                {'params': SelectionFeatureNet.parameters()},
                # {'params': SelectionCriticNet.parameters()},
                {'params': agent.SelectionActorNet.parameters()}
            ], lr=config.learning_rate)
        agent.SelectionMemory = ReplayBuffer(100000)

    # select using rl based on selection epsilon
    for i in tqdm(range(0, config.n_episodes)):

        # check state: (h_action, features)
        h_action, features = [], []
        for n in agents:
            agent = agents[n]
            t = agent.play_times
            if t >= config.h:
                h_action.append(torch.as_tensor(agent.own_memory[t - config.h: t], dtype=torch.float))
                features.append(generate_features(agent, max_reward, max_play_times))
            else:
                break

        if len(h_action) != n_agents:
            # select opponent randomly
            for n in agents:
                m = n
                while m == n:
                    m = random.randint(0, n_agents - 1)
                r1, r2 = env.play(agents[n], agents[m], 1)
        else:
            # process the state
            h_action = torch.stack(h_action, dim=0)
            h_action = h_action.T
            features = torch.stack(features, dim=0)
            features = features.view(-1)

            # select the playing agent randomly and play in sequential way
            prob = np.random.rand(n_agents)
            n = np.argmax(prob)
            player = agents[n]
            state = (h_action[None].to(device), features[None].to(device))
            player.SelectionActorNet.eval()
            state = SelectionFeatureNet(state)
            action_prob = player.SelectionActorNet(state.detach())
            a = torch.distributions.Categorical(action_prob).sample().item()
            m = a+1 if a >= n else a

            # play and optimize PLAY model
            agent1, agent2 = agents[n], agents[m]
            a1, a2 = agent1.act(agent2), agent2.act(agent1)
            _, r1, r2 = env.step(a1, a2)
            env.optimize(agent1, agent2, a1, a2, r1, r2)
            env.update(r1 + r2)

            # process the state and next_state
            h_action, features = [], []
            for n in agents:
                agent = agents[n]
                t = agent.play_times
                if t >= config.h:
                    h_action.append(torch.as_tensor(agent.own_memory[t - config.h: t], dtype=torch.float))
                    features.append(generate_features(agent, max_reward, max_play_times))
            h_action = torch.stack(h_action, dim=0)
            h_action = h_action.T
            features = torch.stack(features, dim=0)
            features = features.view(-1)
            next_state = (h_action[None].to(device), features[None].to(device))
            action = torch.tensor(a1, dtype=torch.int64, device=device)

            # optimize UPDATE model
            SelectionCriticNet.train()
            # SelectionFeatureNet.eval()
            agent1.SelectionActorNet.train()
            values = SelectionCriticNet(state)

            with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
                next_state = SelectionFeatureNet(next_state)
                target = r1 + config.discount * (SelectionCriticNet(next_state))

            # Update critic and actor
            critic_loss = F.mse_loss(values, target)
            advantages = target - values    # loss is measured from error between current and V values
            log_probs, entropy = agent1.SelectionActorNet.evaluate_action(state, action)
            actor_loss = - (log_probs * advantages.detach()).mean()
            # total_loss = (CRITIC_COEF * critic_loss) + actor_loss - (ENTROPY_COEF * entropy)
            CriticOptimizer.zero_grad()
            agent1.SelectionActorNet.zero_grad()
            critic_loss.backward(retain_graph=True)
            actor_loss.backward()
            CriticOptimizer.step()
            agent1.SelectionOptimizer.step()

            for param in SelectionFeatureNet.parameters():
                param.grad.data.clamp_(-1,1)  # DQN gradient clipping: Clamps all elements in input into the range [ min, max ].
            for param in SelectionCriticNet.parameters():
                param.grad.data.clamp_(-1,1)  # DQN gradient clipping: Clamps all elements in input into the range [ min, max ].
            for param in agent1.SelectionActorNet.parameters():
                param.grad.data.clamp_(-1,1)  # DQN gradient clipping: Clamps all elements in input into the range [ min, max ].

            agent1.SelectionMemory.push(state, m, r1, next_state)

            # epsilon decay
            if agent1.config.select_epsilon > agent1.config.min_epsilon:
                agent1.config.select_epsilon *= agent1.config.epsilon_decay

    return agents


    """
    Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
    Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
    Critic_loss = MSE(Q, Q_target)
    Actor_loss = α * log_pi(a|s) - Q(s,a)
    where:
        actor_target(state) -> action
        critic_target(state, action) -> Q-value
    Params
    ======
        experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
        gamma (float): discount factor
    """
