class Environment():
    """
    PD payoff matrix
                Cooperate | Defect
    Cooperate         R,R | S,T
    Defect            T,S | P,P

    R: reward
    P: punishment
    T: temptation
    S: sucker
    T > R > P >S
    2R > T+S
    """
    
    def __init__(self, config: object):
        self.config = config
        self.episode = 0
        self.running_score = 0.0

    def play(self, agent1: object, agent2: object, episodes: int):
        for i in range(episodes):
            a1, a2 = agent1.act(agent2), agent2.act(agent1)
            _, r1, r2 = self.step(a1, a2)
            agent1.update(r1, a1, a2)
            agent2.update(r2, a2, a1)
            agent1.optimize(a1, r1, agent2)
            agent2.optimize(a2, r2, agent1)
        return r1, r2
    
    def step(self, a1: int, a2: int):
        """
        action:
        0 = cooperate
        1 = defect
        """
        episode = self.episode
        self.episode += 1
        assert a1 in [0,1], f"action of agent 1 value is {a1} which not correct"
        assert a2 in [0,1], f"action of agent 2 value is {a2} which not correct"
        if a1==0 and a2==0:
            r1, r2 = self.config.reward, self.config.reward
        elif a1==0 and a2==1:
            r1, r2 = self.config.sucker, self.config.temptation
        elif a1==1 and a2==0:
            r1, r2 = self.config.temptation, self.config.sucker
        elif a1==1 and a2==1:
            r1, r2 = self.config.punishment, self.config.punishment
        
        return episode, r1, r2

    def update(self, reward: int):
        self.running_score = reward + self.config.discount * self.running_score

    def reset(self):
        self.episode = 0
        self.running_score = 0.0