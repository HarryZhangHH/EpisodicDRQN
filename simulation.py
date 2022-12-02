import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from agent import *
from selection import *
from utils import label_encode, argmax, iterate_combination, question, seed_everything
from env import Environment
import sys

def constructAgent(name, config):
    if name == 'A2CLSTM':
        assert 'label' not in config.state_repr, 'you cannot use the label-based state representation method, lstm need the sequential data'
        return ActorCriticLSTMAgent(name, config)
    elif 'Learning' in name:
        return TabularAgent(name, config)
    elif 'DQN' in name:
        return DQNAgent(name, config)
    elif 'LSTM' in name:
        assert 'label' not in config.state_repr, 'you cannot use the label-based state representation method, lstm need the sequential data'
        return LSTMAgent(name, config)
    elif 'A2C' in name:
        return ActorCriticAgent(name, config)
    else:
        return StrategyAgent(name, config)

########################################################################################################################
#################################################### TWO-AGENT GAME ####################################################
########################################################################################################################

def benchmark(strategies, num, config):
    # This benchmark is generated in the geometric setting using the first-visit Monte Carlo method
    # config.n_episode is used as n rounds
    discount = config.discount
    config.discount = 1
    env = Environment(config)
    Q_table_list = []   # for test
    for s in strategies:
        if 'Learning' in strategies[s]:
            continue
        agent1 = constructAgent('MCLearning', config)
        agent2 = constructAgent(strategies[s], config)
        play_times_buffer = []
        a1_running_score_buffer = []
        a2_running_score_buffer = []
        print('You are using the Monte Carlo method')
        print('You opponent uses the strategy ' + strategies[s])
        for i in range(config.n_episodes):
            env.play(agent1, agent2, 1)
            while True:
                prob = torch.rand(1)
                if prob <= discount:
                    env.play(agent1, agent2, 1)
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
        agent1 = constructAgent(strategies[num], config)
        agent2 = constructAgent(strategies[s], config)
        if converge:
            Q_table = agent1.Q_table.clone()
            while True:
                env.play(agent1, agent2, 20*config.h)
                if torch.sum(agent1.Q_table-Q_table) < delta:
                    break
                Q_table = agent1.Q_table.clone()
        else:
            env.play(agent1, agent2, config.n_episodes)
        if 'DQN' in agent1.name or 'LSTM' in agent1.name or 'A2C' in agent1.name:
            print(f'length of loss: {len(agent1.loss)}, average of loss (interval is 2): {np.mean(agent1.loss[::2])}, average of loss (interval is 20): {np.mean(agent1.loss[::20])}, average of loss (interval is 100): {np.mean(agent1.loss[::100])}')
            plt.plot(agent1.loss[::20])
            plt.title(f'agent1: {agent1.name}')
            plt.show()
        if 'DQN' in agent2.name or 'LSTM' in agent2.name or 'A2C' in agent2.name:
            print(f'length of loss: {len(agent2.loss)}, average of loss (interval is 2): {np.mean(agent2.loss[::2])}, average of loss (interval is 20): {np.mean(agent2.loss[::20])}, average of loss (interval is 100): {np.mean(agent2.loss[::100])}')
            plt.plot(agent2.loss[::20])
            plt.title(f'agent:{agent2.name}')
            # plt.show()
        agent1.show()
        agent2.show()
        print("==================================================")
        print(f'{agent1.name} score: {agent1.running_score}\n{agent2.name} score: {agent2.running_score}')
        print("----------------------------------------------------------------------------------------------------------------------------------------------")
        print()
        # print(agent1.Policy_net(torch.tensor([1], dtype=torch.float, device='cpu')), agent1.Policy_net(torch.tensor([0], dtype=torch.float, device='cpu')))
        # if agent2.name == 'DQN':
        #     print(agent2.Policy_net(torch.tensor([1], dtype=torch.float, device='cpu')),
        #           agent2.Policy_net(torch.tensor([0], dtype=torch.float, device='cpu')))


########################################################################################################################
################################################### MULTI-AGENT GAME ###################################################
########################################################################################################################

# multi-agent PD benchmark
MULTI_SELECTION_METHOD = 'LSTM'
def multiAgentSimulate(strategies, config, selection_method=MULTI_SELECTION_METHOD):
    """
    Multi-agent simulation
    Parameters
    ----------
    strategies: dict
        the strategies dictionary
    config: object
    selection_method: string
        selection method: {'RANDOM', 'QLEARNING', 'DQN'}
        separately: RANDOM: select all randomly; RL: using tabular QLEARNING to select; DQN: using DQN to select
         # ALLQ-RANDOM: all agents are Q-agent and select all randomly; FIX-RANDOM: all agents using fix strategies and select all randomly;
         # ALLQ-RL: all agents are Q-agents and use RL to select; FIX-RL: all agents using fix strategies and use RL to select
    """
    # creating an empty list
    lst = []
    lst.append(int(input("Enter number of fix strategy agents : ")))
    lst.append(int(input("Enter number of tabular q-learning agents : ")))
    lst.append(int(input("Enter number of dqn agents : ")))
    lst.append(int(input("Enter number of lstmqn agents : ")))
    # lst.append(int(input("Enter number of a2c agents : ")))
    # lst.append(int(input("Enter number of a2c-lstm agents : ")))

    # construct agents
    seed_everything()
    env = Environment(config)
    agents = {}
    index = 0
    for idx, n in enumerate(lst):
        for _ in range(n):
            if idx == 0:
                agents[index] = constructAgent(strategies[random.randint(0, 6)], config)  # Fix strategy
            if idx == 1:
                agents[index] = constructAgent('QLearning', config)
            if idx == 2:
                agents[index] = constructAgent('DQN', config)
            if idx == 3:
                agents[index] = constructAgent('LSTMQN', config)
            print(f'initialize Agent {index}', end=' ')
            print(agents[index].name)
            index += 1

    # names = locals()
    # for n in range(n_agents):
    #     if 'DQN' in selection_method:
    #         names['n_' + str(n)] = constructAgent('DQN', config)

    if selection_method == 'QLEARNING':
        # Partner selection using tabular q method
        agents = tabular_selection(config, agents, env)

    if selection_method == 'RANDOM':
        agents = random_selection(config, agents, env)

    if selection_method == 'DQN':
        agents = dqn_selection(config, agents, env, False)
    
    if selection_method == 'LSTM':
        agents = dqn_selection(config, agents, env, True)

    for n in range(len(agents)):
        print('Agent{}: name:{}  final score:{}  play time:{}  times to play D:{}  ratio: {}'
            .format(n, agents[n].name, agents[n].running_score,
            len(agents[n].own_memory[:agents[n].play_times]), list(agents[n].own_memory[:agents[n].play_times]).count(1),
                    list(agents[n].own_memory[:agents[n].play_times]).count(1)/len(agents[n].own_memory[:agents[n].play_times])))
    print('The reward for total society: {}'.format(env.running_score/len(agents)))











                








