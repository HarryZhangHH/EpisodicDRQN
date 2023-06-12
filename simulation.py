from agent import *
from selection import *
from utils import *
from component.env import Environment, StochasticGameEnvironment
from model import DQN, DDQN


def construct_agent(name: str, config: object):
    if name == 'A2CLSTM':
        assert 'label' not in config.state_repr, 'you cannot use the label-based state representation method, lstm need the sequential data'
        return ActorCriticLSTMAgent(name, config)
    elif 'Learning' in name:
        return TabularAgent(name, config)
    elif 'DQN' in name:
        return DQNAgent(name, config)
    elif 'LSTM' in name:
        assert 'label' not in config.state_repr, 'you cannot use the label-based state representation method, lstm need the sequential data'
        return LSTMAgent(name, config)
    elif 'A2C' in name:
        return ActorCriticAgent(name, config)
    else:
        return FixStrategyAgent(name, config)

########################################################################################################################
#################################################### TWO-AGENT GAME ####################################################
########################################################################################################################

class TwoAgentSimulation():
    '''
    Implementation of Two agents simulation
    '''
    def __init__(self, config: object) -> None:
        self.config = config

    @staticmethod
    def benchmark_geometric(strategies: dict, num: int, config: object):
        # This benchmark is generated in the geometric setting using the first-visit Monte Carlo method
        # config.n_episode is used as n rounds
        discount = config.discount
        config.discount = 1
        env = Environment(config)
        q_table_list = []   # for test
        for s in strategies:
            if 'Learning' in strategies[s]:
                continue
            agent1 = construct_agent('MCLearning', config)
            agent2 = construct_agent(strategies[s], config)
            play_times_buffer = []
            a1_running_score_buffer = []
            a2_running_score_buffer = []
            print('You are using the Monte Carlo method')
            print('You opponent uses the strategy ' + strategies[s])
            for i in range(config.n_episodes):
                env.play(agent1, agent2, 1)
                while True:
                    prob = torch.rand(1)
                    if prob <= discount:
                        env.play(agent1, agent2, 1)
                    else:
                        agent1.mc_update()
                        break
                play_times_buffer.append(agent1.play_times)
                a1_running_score_buffer.append(agent1.running_score)
                a2_running_score_buffer.append(agent2.running_score)
                # print(f'Playing times: {agent1.play_times}. Discount: {config.discount}')
                # print(f'Your action: {agent2.opponent_memory[:agent2.play_times]}\nOppo action:{agent2.own_memory[:agent2.play_times]}')
                # print(f'Your score: {agent1.running_score}\nOppo score: {agent2.running_score}')
                agent1.reset()
                agent2.reset()
            print(f'The average playing times: {np.mean(play_times_buffer)}, Your average score: {np.mean(a1_running_score_buffer)}, '
                  f'Your opponent average score: {np.mean(a2_running_score_buffer)}')
            if 'Learning' in agent1.name:
                print(f'Your q_table:\n{agent1.q_table}')
            if 'Learning' in agent2.name:
                print(f'Oppo q_table:\n{agent2.q_table}')
            print()
            q_table_list.append(agent1.q_table)
        return q_table_list

    @staticmethod
    def benchmark_episodic(name: str, config: object, thresh: int = 1000, episodic_flag: bool = True, sg_flag: bool = True, method: str = 'DQN', lr_scale: float = 0):
        """
            Two-agent simulation benchmark: Learning Agent vs Fix Agent (Random)
            Parameters
            ----------
            name: str
                learning agent's name
            config: object
            thresh: int
                threshold of the convergence criteria and the length pf the state test set
            episodic_flag: bool
                whether using episodic learning mechanism or not
            sg_flag: bool
                stochastic game or repeated game
            method: str: {'DQN', 'DDQN'}
                play model
            """
        UPDATE_TIMES = 10
        seed_everything()
        print(config.__repr__)
        env = Environment(config) if not sg_flag else StochasticGameEnvironment(config)
        model = DDQN() if method == 'DDQN' else DQN()

        agent1 = construct_agent(name, config)
        agent2 = construct_agent('ALLC', config)

        count = 0
        test_state_list = generate_state(agent1, config.h, config.n_actions, thresh)
        test_q_dict = {'agent1': {}}
        strategy_convergent_episode = 0
        reward_convergent_episode = 0
        last_reward = 0
        converge_agent1_reward = False
        thresh_strategy = thresh * config.min_epsilon + 5
        thresh_network = config.n_episodes
        thresh_reward = 0.01

        while True:
            agent2_pi = np.round(np.random.random(),2)
            transition_episode = np.random.randint(config.batch_size, 5*config.batch_size)
            print(f'Agent 2 Policy: {agent2_pi}, Episode Count: {transition_episode}', end='')
            for i in range(transition_episode):
                if i % transition_episode == 0:
                    if episodic_flag:
                    # agent 1 learns (high learning rate)
                        for _ in range(UPDATE_TIMES):
                            if agent1.memory.__len__() < config.batch_size:
                                continue
                            # random transition batch is taken from experience replay memory
                            transitions = agent1.memory.sample(config.batch_size)
                            batch = agent1.get_batch(transitions)
                            model.train(agent1, batch)
                            agent1.policy.update_epsilon(config)

                        agent1.memory.clean()
                        agent1.target_net.load_state_dict(agent1.policy_net.state_dict())
                        env.reset_state()
                    # else:
                    #     agent1.memory.clean()

                # act and play and optimize
                a1 = agent1.act(agent2)
                sample = np.random.random()
                a2 = 1 if sample <= agent2_pi else 0
                _, r1, r2 = env.step(a1, a2)
                env.optimize(agent1, agent2, a1, a2, r1, r2, flag=not episodic_flag)

            # strategy convergence
            converge_agent1 = agent1.determine_convergence(thresh_strategy, thresh)
            strategy_convergent_episode = agent1.play_times if not converge_agent1 else strategy_convergent_episode
            # print(f' Strategy Convergent: {agent1.play_times}') if converge_agent1 else None

            # reward convergence
            if np.abs(agent1.running_score - last_reward) < thresh_reward:
                converge_agent1_reward = True
            reward_convergent_episode = agent1.play_times if not converge_agent1_reward else reward_convergent_episode
            last_reward = agent1.running_score

            test_q_agent1_list = []
            agent1.policy_net.eval()

            for test_state in test_state_list:
                test_q_agent1 = agent1.policy_net(test_state[None])
                test_q_agent1_list.append(test_q_agent1.to('cpu').detach().numpy())

            agent1.policy_net.train()
            test_q_dict['agent1'][count] = test_q_agent1_list
            # print()
            # print(np.sum(np.diff(test_q_dict['agent1'][count])))
            diff = np.sum(np.diff(test_q_dict['agent1'][count])) - np.sum(np.diff(test_q_dict['agent1'][count-1])) if count > 1 else np.inf
            print(f' Difference: {np.abs(diff)}')
            count += 1

            # network convergence
            if np.abs(diff) < 10 and converge_agent1 is True:
                if not converge_agent1_reward:
                    reward_convergent_episode = np.inf
                break
            if agent1.play_times >= thresh_network:
                break

        # Network convergence
        network_convergent_episode = agent1.play_times if agent1.play_times < thresh_network else None

        for l in test_q_dict['agent1']:
            values, counts = np.unique(np.array(test_q_dict['agent1'][l]), axis=0, return_counts=True)
            print(f" Q value difference: {np.sum(np.diff(test_q_dict['agent1'][l]))}")
            print(l, np.unique(counts, return_counts=True))

        print(f'Playing episodes: {agent1.play_times}')
        agent1.show()
        print(f'Defection ratio: {float(torch.sum(agent1.own_memory)/agent1.play_times)}')
        print(strategy_convergent_episode, network_convergent_episode)
        return agent1, strategy_convergent_episode, network_convergent_episode, reward_convergent_episode

    @staticmethod
    def two_simulate(strategies: dict[int, str], num1: int, num2:int, config: object, delta: float = 0.0001, k: int = 1000):
        converge_flag = False
        alter_flag = False # alternate learning
        seed_everything()
        if num1 >= 7 or num2 >= 7:
            converge_flag = question('Do you want to set the episode to infinity and it will stop automatically when policy converges')
            alter_flag = question('Do you want to apply alternative learning?')
        env = Environment(config)
        print("--------------------------------------------------------------------- GAME ---------------------------------------------------------------------")
        print('You will use the strategy ' + strategies[num1])
        print('You opponent uses the strategy '+strategies[num2])
        env.reset()
        agent1 = construct_agent(strategies[num1], config)
        agent2 = construct_agent(strategies[num2], config)
        converge_agent1 = True
        converge_agent2 = True
        if converge_flag:
            if num1 >= 7:
                converge_agent1 = False
            if num2 >= 7:
                converge_agent2 = False

            thresh = 2 * k * config.min_epsilon
            while True:
                if num1 >= 7 and num2 >=7 and alter_flag:
                    TwoAgentSimulation.two_agent_alter_learning(agent1, agent2, config, env, k)
                else:
                    env.play(agent1, agent2, k)

                converge_agent1 = agent1.determine_convergence(thresh, k)
                converge_agent2 = agent2.determine_convergence(thresh, k)

                print(converge_agent1, converge_agent2)
                if converge_agent1 and converge_agent2:
                    break
                if agent1.play_times >= thresh * k:
                    break

                # q_table = agent1.q_table.clone()
                # while True:
                #     env.play(agent1, agent2, 20*config.h)
                #     if torch.sum(agent1.q_table-q_table) < delta:
                #         break
                #     q_table = agent1.q_table.clone()
            print(f'Convergence: Agent1:{converge_agent1}, Agent2:{converge_agent2}')
        else:
            if num1 >= 7 and num2 >= 7 and alter_flag:
                TwoAgentSimulation.two_agent_alter_learning(agent1, agent2, config, env)
            else:
                env.play(agent1, agent2, config.n_episodes)

        if 'DQN' in agent1.name or 'LSTM' in agent1.name or 'A2C' in agent1.name:
            print(f'length of loss: {len(agent1.loss)}, average of loss (interval is 2): {np.mean(agent1.loss[::2])}, average of loss (interval is 20): {np.mean(agent1.loss[::20])}, average of loss (interval is 100): {np.mean(agent1.loss[::100])}')
            # plt.plot(agent1.loss[::20])
            # plt.title(f'agent1: {agent1.name}')
            # plt.show()
        if 'DQN' in agent2.name or 'LSTM' in agent2.name or 'A2C' in agent2.name:
            print(f'length of loss: {len(agent2.loss)}, average of loss (interval is 2): {np.mean(agent2.loss[::2])}, average of loss (interval is 20): {np.mean(agent2.loss[::20])}, average of loss (interval is 100): {np.mean(agent2.loss[::100])}')
            # plt.plot(agent2.loss[::20])
            # plt.title(f'agent:{agent2.name}')
            # plt.show()
        agent1.show()
        agent2.show()
        print("==================================================")
        print(f'{agent1.name} score: {agent1.running_score}\n{agent2.name} score: {agent2.running_score}')
        print("------------------------------------------------------------------------------------------------------------------------------------------------")
        print()

        # x = [i for i in range(0, agent1.play_times)]
        # plt.figure(figsize=(20, 10))
        # plt.plot(x, agent1.own_memory[0:agent1.play_times], label=agent1.name, alpha=0.5)
        # plt.plot(x, agent2.own_memory[0:agent2.play_times], label=agent2.name, alpha=0.5)
        # plt.legend()
        # plt.ylim(-0.5, 2)
        # plt.xlim(0, agent1.play_times)
        # plt.title(f'agent:{agent1.name} vs agent:{agent2.name}')
        # plt.savefig(f'images/{agent1.name}vs{agent2.name}_result_h={config.h}.png')
        # plt.show()

        # print(agent1.Policy_net(torch.tensor([1], dtype=torch.float, device='cpu')), agent1.Policy_net(torch.tensor([0], dtype=torch.float, device='cpu')))
        # if agent2.name == 'DQN':
        #     print(agent2.Policy_net(torch.tensor([1], dtype=torch.float, device='cpu')),
        #           agent2.Policy_net(torch.tensor([0], dtype=torch.float, device='cpu')))

    @staticmethod
    def two_agent_alter_learning(agent1: object, agent2: object, config: object, env: object, k: int = None, lr_scale: float = 0.001):
        transition_episode = 2*config.batch_size
        n_epsiode = k if k is not None else config.n_episode
        for i in range(n_epsiode):
            if i // transition_episode % 2 == 0:
                # agent 1 learns (high learning rate)
                if i % transition_episode == 0:
                    agent1.memory.clean()
                    agent1.optimizer = torch.optim.Adam(agent1.policy_net.parameters(), lr=config.learning_rate)
                    agent2.optimizer = torch.optim.Adam(agent2.policy_net.parameters(), lr=config.learning_rate*lr_scale)
            else:
                # agent 2 learns (high learning rate)
                if i % transition_episode == 0:
                    agent2.memory.clean()
                    agent1.optimizer = torch.optim.Adam(agent1.policy_net.parameters(), lr=config.learning_rate*lr_scale)
                    agent2.optimizer = torch.optim.Adam(agent2.policy_net.parameters(), lr=config.learning_rate)
            a1, a2 = agent1.act(agent2), agent2.act(agent1)
            _, r1, r2 = env.step(a1, a2)
            env.optimize(agent1, agent2, a1, a2, r1, r2, flag=True)

            if i // transition_episode % 2 == 0:
                # agent 1 learns (high learning rate)
                agent1.policy.update_epsilon(config)
            else:
                # agent 2 learns (high learning rate)
                agent2.policy.update_epsilon(config)


################################################### MULTI-AGENT GAME ###################################################
########################################################################################################################

# multi-agent PD benchmark
MULTI_SELECTION_METHOD = None
def multiAgentSimulate(strategies: dict, config: object, selection_method: str = MULTI_SELECTION_METHOD, thresh: int = 1000):
    """
    Multi-agent simulation
    Parameters
    ----------
    strategies: dict
        the strategies dictionary
    config: object
    selection_method: string
        selection method: {'RANDOM', 'QLEARNING', 'DQN'}
        separately: RANDOM: select all randomly; RL: using tabular QLEARNING to select; DQN: using DQN to select
         # ALLQ-RANDOM: all agents are Q-agent and select all randomly; FIX-RANDOM: all agents using fix strategies and select all randomly;
         # ALLQ-RL: all agents are Q-agents and use RL to select; FIX-RL: all agents using fix strategies and use RL to select
    thresh: int
            threshold of the convergence criteria and the length pf the state test set
    """
    # creating an empty list
    lst = []
    lst.append(int(input("Enter number of fix strategy agents : ")))
    lst.append(int(input("Enter number of dqn agents : ")))
    lst.append(int(input("Enter number of drqn agents : ")))
    # lst.append(int(input("Enter number of tabular q-learning agents : ")))
    # lst.append(int(input("Enter number of lstm-predict agents : ")))
    # lst.append(int(input("Enter number of a2c agents : ")))
    # lst.append(int(input("Enter number of a2c-lstm agents : ")))

    # construct agents
    seed_everything()
    env = Environment(config)
    agents = {}
    index = 0
    with HiddenPrints():
        for idx, n in enumerate(lst):
            for _ in range(n):
                if idx == 0:
                    agents[index] = construct_agent(strategies[random.randint(0, 6)], config)  # Fix strategy
                    # agents[index] = construct_agent('TitForTat', config)
                if idx == 1:
                    agents[index] = construct_agent('DQN', config)
                if idx == 2:
                    agents[index] = construct_agent('LSTMQN', config)
                # if idx == 3:
                #     agents[index] = construct_agent('QLearning', config)
                # if idx == 4:
                #     agents[index] = construct_agent('LSTM', config)

                print(f'initialize Agent {index}', end=' ')
                print(agents[index].name)
                index += 1

    # names = locals()
    # for n in range(n_agents):
    #     if 'DQN' in selection_method:
    #         names['n_' + str(n)] = construct_agent('DQN', config)
    ###################### SIMULTANEOUS ######################
    if selection_method == 'QLEARNING':
        # Partner selection using tabular q method
        agents = tabular_selection(config, agents, env)
    if selection_method == 'RANDOM':
        agents = random_selection(config, agents, env)
    if selection_method == 'DQN':
        agents = dqn_selection(config, agents, env, False)
    if selection_method == 'LSTM':
        agents = dqn_selection(config, agents, env, True)
    if selection_method == 'LSTM-VAR':
        agents = lstm_variant_selection(config, agents, env)

    ###################### SEQUENTIAL #########################
    if selection_method is None:
        sg_flag = question('Do you want to set the environment to stochastic game')
        sg_thresh = None
        if sg_flag:
            print('Set the environment threshold for the stochastic game: {0,1,2,3}')
            sg_thresh = int(input())
            assert sg_thresh in [0,1,2,3], 'you can only set the treshold with 0, 1, 2, 3'
        episodic_flag = question('Do you want to apply EPISODIC learning')
        # agents, select_dict, selected_dict, belief, losses = maxmin_dqrn_selection_play(config, agents, thresh=thresh)
        agents, select_dict, selected_dict, belief, count, convergent_episode, env = maxmin_drqn_selection(config, agents, episodic_flag=episodic_flag, sg_flag=sg_flag, sg_thresh=sg_thresh)
        # agents, select_dict, selected_dict, belief, count, convergent_episode, env = drqn_selection(config, agents)
        print(f'select times: {select_dict}')
        print(f'selected times: {selected_dict}')
        print(f'belief: {np.squeeze(belief)}')

    if selection_method == 'A2C':
        agents = a2c_selection(config, agents, env)

    # show result
    for n in range(len(agents)):
        agents[n].show()

    for n in range(len(agents)):
        print('Agent{}: name:{}  final score:{}  play time:{}  times to play D:{}  ratio: {}  faced D ratio: {}'
              .format(n, agents[n].name, agents[n].running_score,
                      len(agents[n].own_memory[:agents[n].play_times]),
                      list(agents[n].own_memory[:agents[n].play_times]).count(1),
                      list(agents[n].own_memory[:agents[n].play_times]).count(1) / len(agents[n].own_memory[:agents[n].play_times]),
                      list(agents[n].opponent_memory[:agents[n].play_times]).count(1) / len(agents[n].opponent_memory[:agents[n].play_times])))

    print('The reward for total society: {}'.format(env.running_score / len(agents)))

    # plt.figure(figsize=(20, 10))
    # for n in agents:
    #     transitions = agents[n].SelectionMemoryLog.memory
    #     _, actions, rewards, _ = zip(*transitions)
    #     actions = get_index_from_action(np.array(actions, dtype=int), idx)
    #     actions, rewards = np.array(actions), np.array(rewards)
    #     print(f'Agent {n}:', end='')
    #     values, counts = np.unique(actions, return_counts=True)
    #     print(f' opponent_idx: {[i+1 if i >= n else i for i in list(values)]}, counts: {list(counts)} ', end='')
    #     dict_idx = {x: rewards[np.where(actions == x)] for x in values}
    #     print(f'rewards: {[np.mean(y) for _, y in dict_idx.items()]}')
    #
    #     x = [i for i in range(0, agents[n].play_times)]
    #     plt.plot(x, agents[n].own_memory[0:agents[n].play_times], label=agents[n].name+' '+str(n), alpha=0.5)
    # plt.legend()
    # plt.ylim(-0.5, 2)
    # plt.savefig(f'images/maxmin-dqrn.png')
    # plt.show()


def get_index_from_action(action, idx):
    scale = lambda x: x+1 if x>=idx else x
    return list(map(scale, action))









                








