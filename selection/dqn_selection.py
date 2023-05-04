from tqdm import tqdm
import torch.nn as nn

from model import NeuralNetwork, LSTM
from component.memory import UpdateMemory, ReplayBuffer
from utils import *

TARGET_UPDATE = 10
HIDDEN_SIZE = 128
NUM_LAYER = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def dqn_selection(config: object, agents: dict, env: object, rnn: bool = False):
    """
    DQN selection method (benchmark2) - using normal NN or LSTM

    Parameters
    ----------
    config: object
    agents: dict[object]
        dictionary of n unupdated agents
    env: object
    rnn: boolean
        default False: not use LSTM as the function approximator nextwork

    Returns
    -------
    agents: dict[object]
        dictionary of n updated agents
    """
    # construct selection network
    n_agents = len(agents)
    for n in agents:
        agent = agents[n]
        agent.SelectionPolicyNet = NeuralNetwork(n_agents*config.h, n_agents-1, HIDDEN_SIZE).to(device) if not rnn else LSTM(n_agents, HIDDEN_SIZE, NUM_LAYER, n_agents-1).to(device)
        agent.SelectionTargetNet = NeuralNetwork(n_agents*config.h, n_agents-1, HIDDEN_SIZE).to(device) if not rnn else LSTM(n_agents, HIDDEN_SIZE, NUM_LAYER, n_agents-1).to(device)
        agent.SelectionTargetNet.load_state_dict(agent.SelectionPolicyNet.state_dict())
        agent.SelectionMemory = ReplayBuffer(1000)
        agent.SelectionOptimizer = torch.optim.Adam(agent.SelectionPolicyNet.parameters(), lr=config.learning_rate)

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
            state = torch.stack(state, dim=0)
            state = state.view(n_agents*config.h).to(device) if not rnn else state.T.to(device)

            # select opponent based on SelectionNN
            for n in agents:
                # select action by epsilon greedy
                sample = random.random()
                m = n
                if sample > agents[n].config.select_epsilon:
                    agents[n].SelectionPolicyNet.eval()
                    a = int(argmax(agents[n].SelectionPolicyNet(state[None])))
                    m = a+1 if a >= n else a
                else:
                    while m == n:
                        m = random.randint(0, n_agents - 1)
                # play
                agent1, agent2 = agents[n], agents[m]
                a1, a2 = agent1.act(agent2), agent2.act(agent1)
                _, r1, r2 = env.step(a1, a2)
                update_memory.push(n, m, a1, a2, r1, r2, agent1.State.state, agent2.State.state)

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

            # process the next_state
            next_state = []
            for n in agents:
                agent = agents[n]
                t = agent.play_times
                if t >= config.h:
                    next_state.append(torch.as_tensor(agent.own_memory[t - config.h: t], dtype=torch.float))
            next_state = torch.stack(next_state, dim=0)
            next_state = next_state.view(n_agents*config.h).to(device) if not rnn else next_state.T.to(device)

            # push trajectories into Selection ReplayBuffer(Memory) and optimize the model
            losses = []
            for me in update_memory.memory:
                agent1, agent2 = agents[me[0]], agents[me[1]]
                reward = me[4]
                action = me[1]-1 if me[1]>me[0] else me[1]
                agent1.SelectionMemory.push(state, action, reward, next_state)
                agent1.SelectionPolicyNet.train()
                loss = __optimize_model(agent1, n_agents, rnn)
                losses.append(loss) if loss is not None else None
                # Update the target network, copying all weights and biases in DQN
                if agent1.play_times % TARGET_UPDATE == 0:
                    agent1.SelectionTargetNet.load_state_dict(agent1.SelectionPolicyNet.state_dict())

                # epsilon decay
                if agent1.config.select_epsilon > agent1.config.min_epsilon:
                    agent1.config.select_epsilon *= agent1.config.epsilon_decay
        env.update(society_reward)
    return agents

def __optimize_model(agent: object, n_agents: int, rnn: bool):
    """ Train our model """
    # don't learn without some decent experience
    if agent.SelectionMemory.__len__() < agent.config.batch_size:
        return None
    # random transition batch is taken from experience replay memory
    transitions = agent.SelectionMemory.sample(agent.config.batch_size)
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state = zip(*transitions)
    # convert to PyTorch and define types
    state = torch.stack(list(state), dim=0).to(device)
    next_state = torch.stack(list(next_state), dim=0).to(device)
    state = state.view(agent.config.batch_size, -1) if not rnn else state.view(agent.config.batch_size, agent.config.h, n_agents)
    next_state = next_state.view(agent.config.batch_size, -1) if not rnn else next_state.view(agent.config.batch_size, agent.config.h, n_agents)
    action = torch.tensor(action, dtype=torch.int64, device=device)[:, None]  # Need 64 bit to use them as index
    reward = torch.tensor(reward, dtype=torch.float, device=device)[:, None]
    criterion = nn.SmoothL1Loss()
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



