import torch
from agent.abstract_agent import AbstractAgent
from component import Memory, ReplayBuffer

MADTHRESHOLD = 5

class TabularAgent(AbstractAgent):
    """
    Tabular agent including Q-learning agent and Monte Carlo learning agent
    Constructor. Called once at the start of each match.
    This data will persist between rounds of a match but not between matches.
    """
    def __init__(self, name: str, config: object):
        """
        Parameters
        ----------
        name : str
            Learning method
        config : object
            config.h: every agents' most recent h actions are visiable to others which is composed to state
        """
        super(TabularAgent, self).__init__(config)
        # assert 'label' in config.state_repr, 'Note that tabular method can only use the label encoded state representation'
        self.config = config
        self.name = name
        self.h = min(3,config.h)
        print(f'{self.h} previous actions are used')
        self.n_actions = config.n_actions
        self.own_memory = torch.zeros((config.n_episodes * 1000,))
        self.opponent_memory = torch.zeros((config.n_episodes * 1000,))
        self.state = self.StateRepr(method='bilabel', mad_threshold=MADTHRESHOLD)              # an object
        self.q_table = torch.zeros((2 ** self.h * self.state.len(), 2))     # q_table: a tensor (matrix) storing Q values of each state-action pair
        # self.q_table = torch.full((2**config.h, 2), float('-inf'))
        self.play_epsilon = config.play_epsilon
        self.policy = self.EpsilonPolicy(self.q_table, self.play_epsilon, self.config.n_actions)            # an object
        self.memory = ReplayBuffer(10000)

    def act(self, oppo_agent: object):
        """
        Agent act based on the oppo_agent's information
        Parameters
        ----------
        oppo_agent: object

        Returns
        -------
        action index
        """
        # get opponent's last h move
        opponent_h_actions = torch.as_tensor(
            oppo_agent.own_memory[oppo_agent.play_times - self.h: oppo_agent.play_times])
        own_h_actions = torch.as_tensor(
            self.own_memory[self.play_times - self.config.h: self.play_times])
        # label encode
        if self.play_times >= self.h:
            self.state.state = self.state.state_repr(opponent_h_actions, own_h_actions)
        return int(self.__select_action())

    def __select_action(self):
        """ selection action based on epsilon greedy policy """
        a = self.policy.sample_action(self.state.state)
        return a

    def update(self, reward: float, own_action: int, opponent_action: int):
        super(TabularAgent, self).update(reward)
        self.own_memory[self.play_times - 1] = own_action
        self.opponent_memory[self.play_times - 1] = opponent_action
        self.state.oppo_memory = self.opponent_memory[:self.play_times]
        
    def optimize(self, action: int, reward: float, oppo_agent: object, state=None, flag: bool = True):
        super(TabularAgent, self).optimize(action, reward, oppo_agent)
        if self.state.state is None:
            return None
        
        opponent_h_actions = torch.as_tensor(
            oppo_agent.own_memory[oppo_agent.play_times - self.h: oppo_agent.play_times])
        own_h_actions = torch.as_tensor(
            self.own_memory[self.play_times - self.config.h: self.play_times])
        # label encode
        self.state.next_state = self.state.state_repr(opponent_h_actions, own_h_actions)
        self.state.state = self.state.state if state is None else state

        # push the transition into ReplayBuffer
        self.memory.push(self.state.state, action, self.state.next_state, reward)
        if self.name == 'QLearning':
            # Q learning update
            self.q_table[self.state.state, action] = self.q_table[self.state.state, action] + self.config.alpha * \
                                                       (reward + self.config.discount * (torch.max(self.q_table[self.state.next_state])) - self.q_table[self.state.state, action])

    def mc_update(self):
        """ MC update, first-visit, on-policy """
        state_buffer = []
        reward_buffer = list(sub[3] for sub in self.memory.memory)
        for idx, me in enumerate(self.memory.memory):
            state, action, reward = me[0], me[1], me[3]
            if state not in state_buffer:
                G = sum(reward_buffer[idx:])
                self.q_table[state, action] = self.q_table[state, action] + self.config.alpha * \
                                              (G - self.q_table[state, action])
                state_buffer.append(state)

    # def determine_convergence(self, delta:float, q_table: Type.TensorType):
    #     if torch.sum(self.q_table - q_table) < delta:
    #         return True
    #     else:
    #         return False
    def determine_convergence(self, threshold: int, k: int):
        if self.play_times < 2 * k:
            return False
        history_1 = self.own_memory[self.play_times - k: self.play_times]
        history_2 = self.own_memory[self.play_times - 2 * k: self.play_times - k]
        difference = torch.sum(torch.abs(history_1 - history_2))
        if difference > threshold:
            return False
        else:
            return True

    def reset(self):
        """ reset all attribute values expect q_table for episode-end game """
        super(TabularAgent, self).reset()
        self.own_memory = torch.zeros((self.config.n_episodes * 1000,))
        self.opponent_memory = torch.zeros((self.config.n_episodes * 1000,))
        self.play_epsilon = (self.config.play_epsilon + self.play_epsilon)*0.3
        self.state = self.StateRepr(method=self.config.state_repr, mad_threshold=MADTHRESHOLD)
        self.policy = self.EpsilonPolicy(self.q_table, self.play_epsilon, self.config.n_actions)  # an object
        self.memory.clean()


    def show(self):
        print("==================================================")
        start = 0
        if self.play_times > 36:
            start = self.play_times - 36
        print(f'{self.name} play {self.play_times} rounds\nq_table:\n{self.q_table}\nYour action: {self.own_memory[start:self.play_times]}\nOppo action: {self.opponent_memory[start:self.play_times]}')

