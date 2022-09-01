import torch
import random
from agent.fix_strategy_agent import StrategyAgent
from agent.q_learning_agent import QLearningAgent
from utils import decode_one_hot, argmax, iterate_combination
from env import Environment

def play(agent1, agent2, rounds, env):
    for i in range(rounds):
        a1, a2 = agent1.act(agent2), agent2.act(agent1)
        episode, r1, r2 = env.step(a1, a2)
        agent1.update(r1, a1, a2)
        agent2.update(r2, a2, a1)
    return r1, r2

def constructOpponent(name, config):
    if name == 'QLearning':
        return QLearningAgent(config)
    else:
        return StrategyAgent(name, config)

def testStrategy(strategies, num, config):
    # construct env
    env = Environment(config)
    for s in strategies:
        agent1 = StrategyAgent(strategies[num], config)
        print('You opponent uses the strategy '+strategies[s])
        agent2 = constructOpponent(strategies[s], config)
        play(agent1, agent2, config.n_episodes, env)
        print(f'Your action: {agent2.opponent_memory}\nOppo action:{agent2.own_memory}')
        print(f'Your score: {agent1.running_score}\nOpponent score: {agent2.running_score}')

def rlSimulate(strategies, config):
    env = Environment(config)
    for s in strategies:
        print('You opponent uses the strategy '+strategies[s])
        env.reset()
        agent1 = QLearningAgent(config)
        agent2 = constructOpponent(strategies[s], config)
        play(agent1, agent2, config.n_episodes, env)
        agent1.show()
        agent2.show()
        print(f'Your score: {agent1.running_score}\nOppo score: {agent2.running_score}')

def multiSimulate(n_agents, strategies, config, selection_method='RL'):
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
    env = Environment(config)
    names = locals()
    for n in range(n_agents):
        if 'ALLQ' in selection_method:
            s = names['n_' + str(n)] = QLearningAgent(config)
        elif 'FIX' in selection_method:
            s = n%len(strategies)
            names['n_' + str(n)] = constructOpponent(strategies[s], config)
        else:
            s = random.randint(0,len(strategies)-1)
            names['n_' + str(n)] = constructOpponent(strategies[s], config)
        print(f'initialize Agent {n}', end=' ')
        print(names.get('n_' + str(n)).name)
    # select opponent randomly
    for i in range(config.h):
        for n in range(n_agents):
            m = n
            while m == n:
                m = random.randint(0, n_agents-1)
            play(names.get('n_' + str(n)), names.get('n_' + str(m)), 1, env)
            print(n, names.get('n_' + str(n)).name, names.get('n_' + str(n)).running_score, names.get('n_' + str(n)).own_action, end=' ')
            print(m, names.get('n_' + str(m)).name, names.get('n_' + str(m)).running_score, names.get('n_' + str(m)).own_action)
    # select using rl
    for i in range(config.h, config.n_episodes):
        if 'RANDOM' in selection_method:
            for n in range(n_agents):
                m = n
                while m == n:
                    m = random.randint(0, n_agents-1)
                play(names.get('n_' + str(n)), names.get('n_' + str(m)), 1, env)
                print(n, names.get('n_' + str(n)).name, names.get('n_' + str(n)).running_score, len(names.get('n_' + str(n)).own_action), end=' ')
                print(m, names.get('n_' + str(m)).name, names.get('n_' + str(m)).running_score, len(names.get('n_' + str(m)).own_action))
        elif 'RL' in selection_method:
            # initialize Q table
            num = 2**config.h
            state = iterate_combination(num)
            idx = [i for i in range(len(state))]
            Q_table = torch.full((len(state),num), float('-inf'))
            for idx, val in enumerate(state):
                Q_table[idx, list(val)] = 0
            # get history action from agents' memory
            action_hist = torch.zeros((n_agents,config.h))
            for n in range(n_agents):
                t = names.get('n_' + str(n)).play_times
                action_hist[n,:] = torch.as_tensor(names.get('n_' + str(n)).own_memory[t-config.h:t])
            action_hist = decode_one_hot(action_hist.T)
            for n in range(n_agents):
                # select the agent
                state_encode = action_hist[torch.arange(0, action_hist.shape[0]) != n, ...]
                state_encode = torch.unique(state_encode, sorted=True).tolist()
                loc = state.index(tuple(state_encode))
                # select action epsilon greedy
                sample = random.random()
                m = n
                if sample > config.select_epsilon:
                    encode = argmax(Q_table[loc])
                    while m == n:
                        m = action_hist.tolist().index(encode)
                        if type(m) == list:
                            m = random.choice(m)
                else:
                    while m == n:
                        m = random.choice([i for i in range(n_agents)])
                # play
                r1, r2 = play(names.get('n_' + str(n)), names.get('n_' + str(m)), 1, env)
                # update the Q table





                








