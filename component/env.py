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
        self.round = 0
        self.running_score = 0.0
        self.state_count = 1
        print('======= Initialize the environment: Repeated Game =======')

    def play(self, agent1: object, agent2: object, rounds: int):
        for i in range(rounds):
            a1, a2 = agent1.act(agent2), agent2.act(agent1)
            _, r1, r2 = self.step(a1, a2)
            self.optimize(agent1, agent2, a1, a2, r1, r2)
        return r1, r2

    def optimize(self, agent1: object, agent2: object, a1: int, a2: int, r1: float, r2: float, flag: bool = True):
        # can not be called in the multi-agent game
        agent1.update(r1, a1, a2)
        agent2.update(r2, a2, a1)
        agent1.optimize(a1, r1, agent2, flag=flag)
        agent2.optimize(a2, r2, agent1, flag=flag)

    def step(self, a1: int, a2: int):
        """
        Repeated Game
        action:
            0 = cooperate
            1 = defect
        """
        round = self.round
        self.round += 1
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

        return round, r1, r2

    def update_state(self, agents: dict):
        pass

    def update(self, reward: int):
        self.running_score = reward + self.config.discount * self.running_score

    def reset(self):
        self.round = 0
        self.running_score = 0.0

    def reset_state(self):
        pass
    
    @staticmethod
    def evaluate_actions(a1: int, a2: int):
        if a1 == 0 and a2 == 0:
            return 'Mutual Cooperation'
        if a1 == 1 and a2 == 1:
            return 'Mutual Defection'
        if a1 == 0 and a2 == 1:
            return 'Exploitation'
        if a1 == 1 and a2 == 0:
            return 'Deception'
        
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

    def __init__(self, config: object, thresh: int = 0):
        self.config = config
        self.round = 0
        self.running_score = 0.0
        self.s = 1
        self.thresh = thresh
        self.state_count = 2
        print('======= Initialize the environment: Stochastic Game =======')

    @staticmethod
    def check_state(agents: dict, thresh: int = 2):
        state = 0
        for n in agents:
            if agents[n].play_times < thresh:
                return 1
            else:
                state += sum(agents[n].own_memory[agents[n].play_times-thresh : agents[n].play_times])
        return state

    def optimize(self, agent1: object, agent2: object, a1: int, a2: int, r1: float, r2: float, flag: bool = True):
        super(StochasticGameEnvironment, self).optimize(agent1, agent2, a1, a2, r1, r2, flag=flag)
        agents = {}
        agents[0], agents[1] = agent1, agent2
        self.update_state(agents)
        # print(f'game:{self.s}') if self.s == 0 else None

    def update_state(self, agents: dict):
        s = self.check_state(agents)
        # self.s = int(min(s,1))
        self.s = 0 if s<=self.thresh else 1

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
        round = self.round
        self.round += 1
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
        return round, r1, r2

    def reset(self):
        super(StochasticGameEnvironment, self).reset()
        self.reset_state()

    def reset_state(self):
        self.s = 1