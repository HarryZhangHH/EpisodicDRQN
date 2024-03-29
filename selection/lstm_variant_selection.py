from tqdm import tqdm
import torch.nn as nn

from model import LSTMVariant, DQN
from component.memory import UpdateMemory, ReplayBuffer, Memory
from utils import *

TARGET_UPDATE = 10
HIDDEN_SIZE = 256
BATCH_SIZE = 128
FEATURE_SIZE = 3
NUM_LAYER = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def lstm_variant_selection(config: object, agents: dict, env: object):
    """
    DQN-based selection method - using ensemble LSTM (LSTM-VARIANT)

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
    n_agents = len(agents)
    max_reward = config.temptation/(1-config.discount)  # sum of geometric progression
    max_play_times = config.n_episodes * 5

    for n in agents:
        agent = agents[n]
        agent.selection_policy_net = LSTMVariant(n_agents, HIDDEN_SIZE, NUM_LAYER, FEATURE_SIZE*n_agents, n_agents-1, HIDDEN_SIZE).to(device)
        agent.selection_target_net = LSTMVariant(n_agents, HIDDEN_SIZE, NUM_LAYER, FEATURE_SIZE*n_agents, n_agents-1, HIDDEN_SIZE).to(device)
        agent.selection_target_net.load_state_dict(agent.selection_policy_net.state_dict())
        agent.selection_memory = ReplayBuffer(10000)
        agent.selection_optimizer = torch.optim.Adam(agent.selection_policy_net.parameters(), lr=config.learning_rate)
    
    # select using rl based on selection epsilon
    for i in tqdm(range(0, config.n_episodes)):
        society_reward = 0
        update_memory = UpdateMemory(10000)

        # check state: (h_action, features)
        h_action, features = [], []
        for n in agents:
            agent = agents[n]
            t = agent.play_times
            if t >= config.h:
                h_action.append(torch.as_tensor(agent.own_memory[t-config.h : t], dtype=torch.float) )
                features.append(generate_features(agent, max_reward, max_play_times))
            else: break

        if len(h_action) != n_agents:
            # select opponent randomly
            for n in agents:
                m = n
                while m == n:
                    m = random.randint(0, n_agents - 1)
                r1, r2 = env.play(agents[n], agents[m], 1)
                society_reward = society_reward + r1 + r2
        else:
            # process the state
            h_action = torch.stack(h_action, dim=0)
            h_action = h_action.T
            features = torch.stack(features, dim=0)
            features = features.view(-1)
            # select opponent based on SelectionNN
            for n in agents:
                # select action by epsilon greedy
                sample = random.random()
                m = n
                if sample > agents[n].config.select_epsilon:
                    agents[n].selection_policy_net.eval()
                    s = (h_action[None].to(device), features[None].to(device))
                    a = int(argmax(agents[n].selection_policy_net(s)))
                    m = a+1 if a >= n else a
                else:
                    while m == n:
                        m = random.randint(0, n_agents - 1)
                        
                # play
                agent1, agent2 = agents[n], agents[m]
                a1, a2 = agent1.act(agent2), agent2.act(agent1)
                _, r1, r2 = env.step(a1, a2)
                update_memory.push(n, m, a1, a2, r1, r2, agent1.state.state, agent2.state.state)
                # if n ==0 or m == 0:
                #     print('play')
                #     print(f'{i}: {n} {agent1.state.state} {a1} {r1} ')
                #     print(f'{m} {agent2.state.state} {a2} {r2} ')
                #     input()
            
            # update based on the memory
            for me in update_memory.memory:
                agent1, agent2 = agents[me[0]], agents[me[1]]
                a1, a2, r1, r2 = me[2], me[3], me[4], me[5]
                agent1.update(r1, a1, a2)
                agent2.update(r2, a2, a1)
                society_reward = society_reward + r1 + r2

            # optimize the model
            for me in update_memory.memory:
                agent1, agent2 = agents[me[0]], agents[me[1]]
                a1, a2, r1, r2, s1, s2 = me[2], me[3], me[4], me[5], me[6], me[7]
                agent1.optimize(a1, r1, agent2, s1)
                agent2.optimize(a2, r2, agent1, s2)
                # if me[0] ==0 or me[1] == 0:
                #     print('optimize')
                #     print(f'{i}: {me[0]} {s1} {a1} {r1} {agent1.state.next_state}')
                #     print(f'{me[1]} {s2} {a2} {r2} {agent2.state.next_state}')
                #     input()
            
            # process the state and next_state
            state = (h_action.numpy(), features.numpy())
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
            next_state = (h_action.numpy(), features.numpy())

            # push trajectories into Selection ReplayBuffer(Memory) and optimize the model
            losses = []
            for me in update_memory.memory:
                agent1, agent2 = agents[me[0]], agents[me[1]]
                reward = me[4]
                action = me[1]-1 if me[1]>me[0] else me[1]
                agent1.selection_memory.push(state, action, reward, next_state)
                agent1.selection_policy_net.train()
                loss = __optimize_model(agent1, n_agents)
                losses.append(loss) if loss is not None else None

                # update the target network, copying all weights and biases in DQN
                if agent1.play_times % TARGET_UPDATE == 0:
                    agent1.selection_target_net.load_state_dict(agent1.selection_policy_net.state_dict())

                # epsilon decay
                if agent1.config.select_epsilon > agent1.config.min_epsilon:
                    agent1.config.select_epsilon *= agent1.config.epsilon_decay
        env.update(society_reward)
    return agents

def __optimize_model(agent: object, n_agents: int):
    """ Train and optimize our model """
    # don't learn without some decent experience
    if agent.selection_memory.__len__() < BATCH_SIZE:
        return None
    # random transition batch is taken from experience replay memory
    transitions = agent.selection_memory.sample(BATCH_SIZE)
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
    batch = Memory.Transition(state, action, next_state, reward)
    loss = DQN.optimize(agent.selection_policy_net, agent.selection_target_net, agent.selection_optimizer,
                                      batch, agent.config.discount, criterion)
    return loss.item()