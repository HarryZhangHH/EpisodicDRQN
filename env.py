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
    
    def __init__(self, config):
        self.config = config
        self.episode = 0
    
    def step(self, a1, a2):
        """
        action:
        0 = cooperate
        1 = defect
        """
        episode = self.episode
        self.episode += 1

        if a1==0 and a2==0:
            r1, r2 = self.config.reward, self.config.reward
        elif a1==0 and a2==1:
            r1, r2 = self.config.sucker, self.config.temptation
        elif a1==1 and a2==0:
            r1, r2 = self.config.temptation, self.config.sucker
        elif a1==1 and a2==1:
            r1, r2 = self.config.punishment, self.config.punishment
        
        return episode, r1, r2
