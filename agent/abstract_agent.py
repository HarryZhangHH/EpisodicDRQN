class AbstractAgent():
    """ 
    Abstract an agent (superclass)
    Constructor. Called once at the start of each match.
    This data will persist between rounds of a match but not between matches.
    """

    def __init__(self, config):
        self.config = config
        self.running_score = 0.0
    
    def act(self):
        pass

    def update(self, reward):
        self.running_score = reward + self.config.discount * self.running_score

    """
    Process the results of a round. This provides an opportunity to store data 
    that preserves the memory of previous rounds.

    Parameters
    ----------
    my_strategy: bool
    other_strategy: bool
    """
    def process_results(self, my_strategy, other_strategy):
        pass