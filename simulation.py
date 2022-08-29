import torch
import random
from agent.fix_strategy_agent import StrategyAgent
from agent.q_learning_agent import QLearningAgent, decode_one_hot
from env import Environment


def play(agent1, agent2, rounds, env):
    for i in range(rounds):
        a1, a2 = agent1.act(agent2), agent2.act(agent1)
        episode, r1, r2 = env.step(a1, a2)
        agent1.update(r1, a1, a2, agent2)
        agent2.update(r2, a2, a1, agent1)
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

def multiSimulate(n_agents, strategies,config):
    env = Environment(config)
    names = locals()
    for n in range(n_agents):
        s = random.randint(1,len(strategies))
        if strategies[s] == 'QLearning':
            names['n_' + str(n) ] = QLearningAgent(config)
        else:
            names['n_' + str(n) ] = StrategyAgent(strategies[s], config)
        print(f'initialize Agent {n}', end=' ')
        print(names.get('n_' + str(n)).name)  
    # select opponent randomly
    for i in range(config.h):
        for n in range(n_agents):
            m = n
            while m == n:
                m = random.randint(0, n_agents-1)
            play(names.get('n_' + str(n)), names.get('n_' + str(m)), 1, env)
            print(n, names.get('n_' + str(n)).name, names.get('n_' + str(n)).own_memory, names.get('n_' + str(n)).opponent_memory, end=' ')
            print(m, names.get('n_' + str(m)).name, names.get('n_' + str(m)).own_memory, names.get('n_' + str(m)).opponent_memory)
    # select using rl
    for i in range(config.h, config.n_episodes):
        state = torch.zeros((n_agents,config.h))
        for n in range(n_agents):
            i = names.get('n_' + str(n)).play_times
            state[n,:] = torch.as_tensor(names.get('n_' + str(n)).own_memory[i-config.h:i])
        for n in range(n_agents):
            new_state = state[torch.arange(0, state.shape[0]) != n, ...]
            # select_opponent(new_state)


