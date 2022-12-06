import random
from tqdm import tqdm

def random_selection(config: object, agents: dict[object], env: object):
    n_agents = len(agents)
    for i in tqdm(range(0, config.n_episodes)):
        society_reward = 0
        for n in range(n_agents):
            m = n
            while m == n:
                m = random.randint(0, n_agents-1)
            r1, r2 = env.play(agents[n], agents[m], 1)
            society_reward = society_reward + r1 + r2
            if i < 5:
                print(f"{i} episode")
                print(f"Agent {n}, {agents[n].name}, score: {agents[n].running_score}, play times: {agents[n].play_times}", end=' ')
                print("  v.s.  ", end='')
                print(f"Agent {m}, {agents[m].name}, score: {agents[m].running_score}, play times: {agents[m].play_times}")
        env.update(society_reward)
    return agents