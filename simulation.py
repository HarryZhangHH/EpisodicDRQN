import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from agent.fix_strategy_agent import StrategyAgent
from agent.tabluar_agent import TabularAgent, SelectMemory
from agent.dqn_agent import DQNAgent
from agent.lstm_agent import LSTMAgent
from utils import label_encode, argmax, iterate_combination, question, seed_everything
from env import Environment
import sys

def play(agent1, agent2, rounds, env):
    for i in range(rounds):
        a1, a2 = agent1.act(agent2), agent2.act(agent1)
        episode, r1, r2 = env.step(a1, a2)
        agent1.update(r1, a1, a2)
        agent2.update(r2, a2, a1)
    return r1, r2

def constructOpponent(name, config):
    if 'Learning' in name:
        return TabularAgent(name, config)
    elif 'DQN' in name:
        return DQNAgent(name, config)
    elif 'LSTM' in name:
        return LSTMAgent(name, config)
    else:
        return StrategyAgent(name, config)

def benchmark(strategies, num, config):
    # This benchmark is generated in the geometric setting using the first-visit Monte Carlo method
    discount = config.discount
    config.discount = 1
    env = Environment(config)
    Q_table_list = []   # for test
    for s in strategies:
        if 'Learning' in strategies[s]:
            continue
        agent1 = constructOpponent('MCLearning', config)
        agent2 = constructOpponent(strategies[s], config)
        play_times_buffer = []
        a1_running_score_buffer = []
        a2_running_score_buffer = []
        print('You are using the Monte Carlo method')
        print('You opponent uses the strategy ' + strategies[s])
        for i in range(config.n_episodes):
            play(agent1, agent2, 1, env)
            while True:
                prob = torch.rand(1)
                if prob <= discount:
                    play(agent1, agent2, 1, env)
                else:
                    agent1.mc_update()
                    break
            play_times_buffer.append(agent1.play_times)
            a1_running_score_buffer.append(agent1.running_score)
            a2_running_score_buffer.append(agent2.running_score)
            # print(f'Playing times: {agent1.play_times}. Discount: {config.discount}')
            # print(f'Your action: {agent2.opponent_memory[:agent2.play_times]}\nOppo action:{agent2.own_memory[:agent2.play_times]}')
            # print(f'Your score: {agent1.running_score}\nOppo score: {agent2.running_score}')
            agent1.reset()
            agent2.reset()
        print(f'The average playing times: {np.mean(play_times_buffer)}, Your average score: {np.mean(a1_running_score_buffer)}, '
              f'Your opponent average score: {np.mean(a2_running_score_buffer)}')
        if 'Learning' in agent1.name:
            print(f'Your Q_table:\n{agent1.Q_table}')
        if 'Learning' in agent2.name:
            print(f'Oppo Q_table:\n{agent2.Q_table}')
        print()
        Q_table_list.append(agent1.Q_table)
    return Q_table_list

def twoSimulate(strategies, num, config, delta = 0.0001):
    converge = False
    seed_everything()
    if 'Learning' in strategies[num]:
        converge = question('Do you want to set the episode to infinity and it will stop automatically when policy converges')
    env = Environment(config)
    for s in strategies:
        print("---------------------------------------------------------------------GAME---------------------------------------------------------------------")
        print('You will use the strategy ' + strategies[num])
        print('You opponent uses the strategy '+strategies[s])
        env.reset()
        agent1 = constructOpponent(strategies[num], config)
        agent2 = constructOpponent(strategies[s], config)
        if converge:
            Q_table = agent1.Q_table.clone()
            while True:
                play(agent1, agent2, 20*config.h, env)
                if torch.sum(agent1.Q_table-Q_table) < delta:
                    break
                Q_table = agent1.Q_table.clone()
        else:
            play(agent1, agent2, config.n_episodes, env)
        if 'DQN' in agent1.name or 'LSTM' in agent1.name:
            print(len(agent1.loss), np.mean(agent1.loss[::2]), np.mean(agent1.loss[::20]), np.mean(agent1.loss[::100]))
            plt.plot(agent1.loss[::20])
            plt.title(f'agent1: {agent1.name}')
            plt.show()
        if 'DQN' in agent2.name or 'LSTM' in agent2.name:
            print(len(agent2.loss), np.mean(agent2.loss[::2]), np.mean(agent2.loss[::20]), np.mean(agent2.loss[::100]))
            plt.plot(agent2.loss[::20])
            plt.title(f'agent:{agent2.name}')
            plt.show()
        agent1.show()
        agent2.show()
        print(f'Your score: {agent1.running_score}\nOppo score: {agent2.running_score}')
        print("----------------------------------------------------------------------------------------------------------------------------------------------")
        print()
        # print(agent1.Policy_net(torch.tensor([1], dtype=torch.float, device='cpu')), agent1.Policy_net(torch.tensor([0], dtype=torch.float, device='cpu')))
        # if agent2.name == 'DQN':
        #     print(agent2.Policy_net(torch.tensor([1], dtype=torch.float, device='cpu')),
        #           agent2.Policy_net(torch.tensor([0], dtype=torch.float, device='cpu')))


# benchmark
def multiBenchmark(n_agents, strategies, config, selection_method='ALLQRL'):
    """
    Multi-agent simulation
    Parameters
    ----------
    n_agents: int
        # of agents
    strategies: dict
        the strategies dictionary
    config: object
    seletion: string
        selection method: {'RANDOM', 'ALLQRANDOM', 'FIXRANDOM', 'RL', 'ALLQRL', 'FIXRL'}
        separately: select all randomly; all agents are q-agent and select all randomly; fix agents and select all randomly
                    using RL to select, all agents are q-agents and use RL to select; fix agents and use RL to select
    """
    # construct agents
    env = Environment(config)
    names = locals()
    for n in range(n_agents):
        if 'ALLQ' in selection_method:
            s = names['n_' + str(n)] = TabularAgent('QLearning', config)
        elif 'FIX' in selection_method:
            s = n%len(strategies)
            names['n_' + str(n)] = constructOpponent(strategies[s], config)
        else:
            s = random.randint(0,len(strategies)-1)
            names['n_' + str(n)] = constructOpponent(strategies[s], config)
        print(f'initialize Agent {n}', end=' ')
        print(names.get('n_' + str(n)).name)
    
    # initialize Q table
    num = 2**config.h
    state_list = iterate_combination(num)
    Q_table = torch.full((len(state_list),num), float('-inf'))
    for idx, val in enumerate(state_list):
        Q_table[idx, list(val)] = 0

    # select opponent randomly
    for i in range(config.h):
        society_reward = 0
        for n in range(n_agents):
            m = n
            while m == n:
                m = random.randint(0, n_agents-1)
            r1, r2 = play(names.get('n_' + str(n)), names.get('n_' + str(m)), 1, env)
            society_reward = society_reward + r1 + r2
            print(n, names.get('n_' + str(n)).name, names.get('n_' + str(n)).running_score, names.get('n_' + str(n)).own_action, end=' ')
            print(m, names.get('n_' + str(m)).name, names.get('n_' + str(m)).running_score, names.get('n_' + str(m)).own_action)
        env.update(society_reward)
    # select using rl
    for i in range(config.h, config.n_episodes):
        society_reward = 0
        if 'RANDOM' in selection_method:
            for n in range(n_agents):
                m = n
                while m == n:
                    m = random.randint(0, n_agents-1)
                r1, r2 = play(names.get('n_' + str(n)), names.get('n_' + str(m)), 1, env)
                society_reward = society_reward + r1 + r2
                print(n, names.get('n_' + str(n)).name, names.get('n_' + str(n)).running_score, len(names.get('n_' + str(n)).own_action), end=' ')
                print(m, names.get('n_' + str(m)).name, names.get('n_' + str(m)).running_score, len(names.get('n_' + str(m)).own_action))
        elif 'RL' in selection_method:
            memory = SelectMemory(10000)
            # get history action from agents' memory
            action_hist = torch.zeros((n_agents,config.h))
            for n in range(n_agents):
                t = names.get('n_' + str(n)).play_times
                action_hist[n,:] = torch.as_tensor(names.get('n_' + str(n)).own_memory[t-config.h:t])
            action_hist = label_encode(action_hist.T)
            for n in range(n_agents):
                # select the agent
                state_encode = action_hist[torch.arange(0, action_hist.shape[0]) != n, ...]
                state_encode = tuple(torch.unique(state_encode, sorted=True).tolist())
                state = state_list.index(state_encode)

                # select action by epsilon greedy
                sample = random.random()
                m = n
                if i >= (2**config.h)*config.n_actions and sample > config.select_epsilon:
                    action_encode = argmax(Q_table[state])
                    while m == n:
                        m = [i for i, x in enumerate(action_hist.tolist()) if x == action_encode]
                        m = random.choice(m)
                else:
                    while m == n:
                        m = random.randint(0, n_agents-1)
                    action_encode = action_hist[m]

                # play
                agent1, agent2 = names.get('n_' + str(n)), names.get('n_' + str(m))
                a1, a2 = agent1.act(agent2), agent2.act(agent1)
                episode, r1, r2 = env.step(a1, a2)
                # store the data into the select buffer and update all the Q_table after all agents play
                # Agent = namedtuple('Agent', ['state', 'action', 'agent_1', 'agent_2', 'action_1', 'action_2', 'reward_1', 'reward_2'])
                memory.push(state, action_encode, n, m, a1, a2, r1, r2)

            # update the Q table
            for me in memory.memory:
                agent1, agent2 = names.get('n_' + str(me[2])), names.get('n_' + str(me[3]))
                a1, a2, r1, r2 = me[4], me[5], me[6], me[7]
                agent1.update(r1, a1, a2)
                agent2.update(r2, a2, a1)
                society_reward = society_reward + r1 + r2

            # get history action from agents' memory
            action_hist = torch.zeros((n_agents,config.h))
            for n in range(n_agents):
                t = names.get('n_' + str(n)).play_times
                action_hist[n,:] = torch.as_tensor(names.get('n_' + str(n)).own_memory[t-config.h:t])
            action_hist = label_encode(action_hist.T)

            for me in memory.memory:
                state, action, reward, agent_idx = me[0], me[1], me[6], me[3]
                next_state = action_hist[torch.arange(0, action_hist.shape[0]) != agent_idx, ...]
                Q_table[state, action] = (1-config.alpha)*Q_table[state, action] \
                    + config.alpha*(reward + config.discount*(torch.max(Q_table[next_state])))
        env.update(society_reward)
    print(Q_table)
    for n in range(n_agents):
        print('Agent{}: name:{} final score:{} play time:{} times to play D:{}'
            .format(n, names.get('n_' + str(n)).name, names.get('n_' + str(n)).running_score, 
            len(names.get('n_' + str(n)).own_memory[:names.get('n_' + str(n)).play_times]), list(names.get('n_' + str(n)).own_memory[:names.get('n_' + str(n)).play_times]).count(1)))
    print('The reward for total society: {}'.format(env.running_score))




                








                








