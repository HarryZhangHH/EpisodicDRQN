from agent.fix_strategy_agent import StrategyAgent
from env import Environment


def play(agent1, agent2, rounds, env):
    for i in range(rounds):
        a1, a2 = agent1.act(), agent2.act()
        _, r1, r2 = env.step(a1, a2)
        agent1.update(r1, a2)
        agent2.update(r2, a1)
    print(f'Your action: {agent2.opponent_memory}\nOppo action:{agent1.opponent_memory}')
    

def testStrategy(strategies, num, config):
    # construct env
    env = Environment(config)
    for s in strategies:
        agent1 = StrategyAgent(strategies[num], config)
        print('You opponent uses the strategy '+strategies[s])
        agent2 = StrategyAgent(strategies[s], config)
        play(agent1, agent2, config.n_episodes, env)
        print(f'Your score: {agent1.running_score}\nOpponent score: {agent2.running_score}')
