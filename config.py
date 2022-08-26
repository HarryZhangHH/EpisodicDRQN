class Config():
    def __init__(self, config):
        self.parse_config(**config)
    
    def parse_config(self, reward, sucker, temptation, punishment, n_episodes, discount, play_epsilon):
        # game payoffs
        self.reward = reward
        self.sucker = sucker
        self.temptation = temptation
        self.punishment = punishment
        self.n_episodes = n_episodes
        self.discount = discount
        self.play_epsilon = play_epsilon

    def __repr__(self):
        return 'Configs: ' + ' episodes=' + str(self.n_episodes) + \
            ' discount=' + str(self.discount) + \
            '\npayoff matrix: ' + \
            ' r=' + str(self.reward) + \
            ' t=' + str(self.temptation) + \
            ' s=' + str(self.sucker) + \
            ' p=' + str(self.punishment) + \
            ' epsilon=' + str(self.play_epsilon)
