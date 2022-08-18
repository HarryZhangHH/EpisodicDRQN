from agent.fix_strategy_agent import StrategyAgent
from env import Environment


def play(agent1, agent2, rounds, env):
    for i in range(rounds):
        a1, a2 = agent1.act(), agent2.act()
        print(a1, type(a1), a2)
        _, r1, r2 = env.step(a1, a2)
        agent1.update(r1, a2)
        agent2.update(r2, a1)
    
def testStrategy(strategies, num, config):
    # construct env
    env = Environment(config)
    agent1 = StrategyAgent(strategies[num], config)
    for s in strategies:
        agent2 = StrategyAgent(strategies[s], config)
        play(agent1, agent2, config.n_episodes, env)
        print(agent1.running_score, agent2.running_score)
