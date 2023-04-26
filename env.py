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
            self.optimize(agent1, agent2, a1, a2, r1, r2)
            # agent1.Policy.update_epsilon(self.config)
            # agent2.Policy.update_epsilon(self.config)
        return r1, r2

    def optimize(self, agent1: object, agent2: object, a1: int, a2: int, r1: float, r2: float, flag: bool = True):
        # can not be called in the multi-agent game
        agent1.update(r1, a1, a2)
        agent2.update(r2, a2, a1)
        agent1.optimize(a1, r1, agent2, flag=flag)
        agent2.optimize(a2, r2, agent1, flag=flag)

    def step(self, a1: int, a2: int, sg_flag: bool = False):
        """
        Repeated Game
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

        if sg_flag:
            # stag hunt
            if a1 == 0 and a2 == 0:
                r1, r2 = self.config.temptation, self.config.temptation
            elif a1 == 0 and a2 == 1:
                r1, r2 = self.config.sucker, self.config.reward
            elif a1 == 1 and a2 == 0:
                r1, r2 = self.config.reward, self.config.sucker
            elif a1 == 1 and a2 == 1:
                r1, r2 = self.config.punishment, self.config.punishment
        return episode, r1, r2

    def update(self, reward: int):
        self.running_score = reward + self.config.discount * self.running_score

    def reset(self):
        self.episode = 0
        self.running_score = 0.0


class StochasticGameEnvironment(Environment):
    """
    PD payoff matrix
                Cooperate | Defect
    Cooperate         R,R | S,T
    Defect            T,S | P,P

    R: reward
    P: punishment
    T: temptation
    S: sucker

    Stochastic Game
        state:
            1 = prisoner's dilemma
            0 = stag hunt
    """

    def __init__(self, config: object):
        super(StochasticGameEnvironment, self).__init__(config)
        self.s = 1

    @staticmethod
    def check_state(agent1: object, agent2: object):
        if agent1.play_times >= agent1.config.h and agent2.play_times >= agent2.config.h:
            state = sum(agent1.own_memory[agent1.play_times - agent1.config.h: agent1.play_times]) + sum(agent2.own_memory[agent2.play_times - agent2.config.h: agent2.play_times])
            return int(min(state,1))
        else: return 1

    def optimize(self, agent1: object, agent2: object, a1: int, a2: int, r1: float, r2: float, flag: bool = True):
        super(StochasticGameEnvironment, self).optimize(agent1, agent2, a1, a2, r1, r2, flag=flag)
        s = self.check_state(agent1, agent2)
        self.s = s
        # print(f'game:{self.s}') if self.s == 0 else None

    def step(self, a1: int, a2: int):
        """
        Stochastic Game
        state:
            1 = prisoner's dilemma
            0 = stag hunt
        action:s
            0 = cooperate
            1 = defect
        """
        episode = self.episode
        self.episode += 1
        assert a1 in [0, 1], f"action of agent 1 value is {a1} which not correct"
        assert a2 in [0, 1], f"action of agent 2 value is {a2} which not correct"

        if self.s == 1:
            if a1 == 0 and a2 == 0:
                r1, r2 = self.config.reward, self.config.reward
            elif a1 == 0 and a2 == 1:
                r1, r2 = self.config.sucker, self.config.temptation
            elif a1 == 1 and a2 == 0:
                r1, r2 = self.config.temptation, self.config.sucker
            elif a1 == 1 and a2 == 1:
                r1, r2 = self.config.punishment, self.config.punishment

        if self.s == 0:
            if a1 == 0 and a2 == 0:
                r1, r2 = self.config.temptation, self.config.temptation
            elif a1 == 0 and a2 == 1:
                r1, r2 = self.config.sucker, self.config.reward
            elif a1 == 1 and a2 == 0:
                r1, r2 = self.config.reward, self.config.sucker
            elif a1 == 1 and a2 == 1:
                r1, r2 = self.config.punishment, self.config.punishment

        return episode, r1, r2

    def reset(self):
        super(StochasticGameEnvironment, self).reset()
        self.s = 1