import torch
import random
from agent.fix_strategy_agent import StrategyAgent
from agent.q_learning_agent import QLearningAgent, decode_one_hot
from env import Environment


def play(agent1, agent2, rounds, env):
    for i in range(rounds):
        a1, a2 = agent1.act(), agent2.act()
        episode, r1, r2 = env.step(a1, a2)
        agent1.update(r1, a1, a2, episode)
        agent2.update(r2, a2, a1, episode)

def testStrategy(strategies, num, config):
    # construct env
    env = Environment(config)
    for s in strategies:
        agent1 = StrategyAgent(strategies[num], config)
        print('You opponent uses the strategy '+strategies[s])
        agent2 = StrategyAgent(strategies[s], config)
        play(agent1, agent2, config.n_episodes, env)
        print(f'Your action: {agent2.opponent_memory}\nOppo action:{agent2.own_memory}')
        print(f'Your score: {agent1.running_score}\nOpponent score: {agent2.running_score}')

def rlSimulate(strategies, config):
    env = Environment(config)
    for s in strategies:
        print('You opponent uses the strategy '+strategies[s])
        env.reset()
        agent1 = QLearningAgent(config)
        agent2 = StrategyAgent(strategies[s], config)
        for i in range(config.n_episodes):
            # the last h actions of the opponent
            state = decode_one_hot(agent1.opponent_memory[i-1:i+agent1.h-1])
            # state = torch.cat([agent1.own_memory[i:i+agent1.h], agent1.opponent_memory[i:i+agent1.h]], dim=1)
            if i == 0:
                # initialize state
                a1 = random.randrange(agent1.n_actions)
            elif i <= 2*(2**agent1.h)*agent1.n_actions:
                a1 = int(agent1.select_action(state, True))
            else:
                a1 = int(agent1.select_action(state))
            a2 = agent2.act()
            episode, r1, r2 = env.step(a1, a2)
            agent1.update(r1, a1, a2, episode)
            agent2.update(r2, a2, a1, episode)
        agent1.show()
        print(f'Your score: {agent1.running_score}\nOppo score: {agent2.running_score}')


