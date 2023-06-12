import numpy
import random
import torch
import torch.nn as nn
from utils import *
from model import *
from simulation import *
from component import *
from tqdm import tqdm
import seaborn as sns
from skimage.color import rgb2gray

IMAGE_SIZE = 5
INPUT_IMAGE_DIM = 21
OUT_SIZE = 4
MEMORY_SIZE = 100000
TARGET_UPDATE_TIMES = 10000
UPDATE_TIMES = 5
HIDDEN_SIZE = 128
LEARNING_RATE = 0.0005
EPSILON_DECAY = 0.9999
MIN_EPSILON = 0.1
N_STEPS = 20000
TOTAL_EPISODES = 50
SETTLEMENT_PROB = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def simulate(config: object, seed: int = 42, method: str = 'normal', total_episodes: float = TOTAL_EPISODES):
    seed_everything(seed)
    set_hyper_parameter(config)
    env = Coins(size=IMAGE_SIZE, config=config, image_dim=INPUT_IMAGE_DIM)

    agent1 = initialize_agent('agent1', config)
    agent2 = initialize_agent('agent2', config)

    fill_memory(agent1, agent2, env, config)

    if method == 'normal':
        for ep in tqdm(range(total_episodes)):
            normal_drqn(agent1, agent2, env, config, total_episodes=total_episodes)

    if method == 'episodic':
        # optimize the policy network UPDATE_TIMES times
        for _ in range(int(UPDATE_TIMES * 10)):
            optimize(agent1)
            optimize(agent2)
        # update the target network, copying all weights and biases in DRQN
        agent1.target_net.load_state_dict(agent1.policy_net.state_dict())
        agent2.target_net.load_state_dict(agent2.policy_net.state_dict())
        for ep in tqdm(range(total_episodes)):
            episodic_drqn(agent1, agent2, env, config, total_episodes=total_episodes)


def normal_drqn(agent1: object, agent2: object, env: object, config: object, total_episodes: float = TOTAL_EPISODES):
    state = env.reset()
    processed_state = preprocess_image(state)
    agent1.temporary_memory.clean()
    agent2.temporary_memory.clean()
    synchronize_position(agent1, env)
    synchronize_position(agent2, env)

    agent1_hs, agent1_cs = agent1.policy_net.init_hidden_states(batch_size=1)
    agent2_hs, agent2_cs = agent2.policy_net.init_hidden_states(batch_size=1)

    for _ in range(config.n_episodes):

        with torch.no_grad():
            s = torch.from_numpy(processed_state).float().to(device)
            agent1_q_vals, (agent1_hs, agent1_cs) = agent1.policy_net(s, batch_size=1, time_step=1,
                                                                      hidden_state=agent1_hs, cell_state=agent1_cs)
            agent2_q_vals, (agent2_hs, agent2_cs) = agent2.policy_net(s, batch_size=1, time_step=1,
                                                                      hidden_state=agent2_hs, cell_state=agent2_cs)

        a1 = agent1.policy.sample_action(None, agent1_q_vals)
        a2 = agent2.policy.sample_action(None, agent2_q_vals)

        r1, r2, next_state = env.step(agent1, agent2, a1, a2)
        processed_next_state = preprocess_image(next_state)
        agent1.update(r1, a1, a2)
        agent2.update(r2, a2, a1)
        env.update(r1 + r2)

        agent1.temporary_memory.push(processed_state, a1, processed_next_state, r1)
        agent2.temporary_memory.push(processed_state, a2, processed_next_state, r2)

        if agent1.temporary_memory.__len__() >= agent1.config.h:
            store_transition_in_buffer(agent1.temporary_memory, agent1.config.h, agent1.memory)

        if agent2.temporary_memory.__len__() >= agent2.config.h:
            store_transition_in_buffer(agent2.temporary_memory, agent2.config.h, agent2.memory)

        state = next_state
        processed_state = processed_next_state

        # optimize
        if agent1.play_times % UPDATE_TIMES == 0:
            optimize(agent1)
        if agent2.play_times % UPDATE_TIMES == 0:
            optimize(agent2)

        if agent1.play_times % TARGET_UPDATE_TIMES == 0:
            agent1.target_net.load_state_dict(agent1.policy_net.state_dict())
        if agent2.play_times % TARGET_UPDATE_TIMES == 0:
            agent2.target_net.load_state_dict(agent2.policy_net.state_dict())

    if agent1.policy.epsilon > agent1.config.min_epsilon:
        agent1.policy.epsilon -= (1 - agent1.config.min_epsilon) / total_episodes
    if agent1.policy.epsilon < agent1.config.min_epsilon:
        agent1.policy.epsilon = agent1.config.min_epsilon
    if agent2.policy.epsilon > agent2.config.min_epsilon:
        agent2.policy.epsilon -= (1 - agent2.config.min_epsilon) / total_episodes
    if agent2.policy.epsilon < agent2.config.min_epsilon:
        agent2.policy.epsilon = agent2.config.min_epsilon

def episodic_drqn(agent1: object, agent2: object, env: object, config: object, settlement_prob: float = SETTLEMENT_PROB, total_episodes: int = TOTAL_EPISODES, update_times : int = UPDATE_TIMES**2):
    state = env.reset()
    processed_state = preprocess_image(state)
    agent1.temporary_memory.clean()
    agent2.temporary_memory.clean()
    synchronize_position(agent1, env)
    synchronize_position(agent2, env)

    agent1_hs, agent1_cs = agent1.policy_net.init_hidden_states(batch_size=1)
    agent2_hs, agent2_cs = agent2.policy_net.init_hidden_states(batch_size=1)

    for _ in range(config.n_episodes):

        if np.random.rand(1) < settlement_prob:
            # settlement and update network
            if agent1.memory.__len__() > UPDATE_TIMES*agent1.config.batch_size and agent2.memory.__len__() > UPDATE_TIMES*agent2.config.batch_size:
                # optimize the policy network UPDATE_TIMES times
                for _ in range(update_times):
                    optimize(agent1)
                    optimize(agent2)
                # update the target network, copying all weights and biases in DRQN
                agent1.target_net.load_state_dict(agent1.policy_net.state_dict())
                agent2.target_net.load_state_dict(agent2.policy_net.state_dict())
                # clean the buffer
                agent1.memory.clean()
                agent2.memory.clean()
                agent1.temporary_memory.clean()
                agent2.temporary_memory.clean()
            continue

        with torch.no_grad():
            s = torch.from_numpy(processed_state).float().to(device)
            agent1_q_vals, (agent1_hs, agent1_cs) = agent1.policy_net(s, batch_size=1, time_step=1,
                                                                      hidden_state=agent1_hs, cell_state=agent1_cs)
            agent2_q_vals, (agent2_hs, agent2_cs) = agent2.policy_net(s, batch_size=1, time_step=1,
                                                                      hidden_state=agent2_hs, cell_state=agent2_cs)

        a1 = agent1.policy.sample_action(None, agent1_q_vals)
        a2 = agent2.policy.sample_action(None, agent2_q_vals)

        r1, r2, next_state = env.step(agent1, agent2, a1, a2)
        processed_next_state = preprocess_image(next_state)
        agent1.update(r1, a1, a2)
        agent2.update(r2, a2, a1)
        env.update(r1 + r2)

        if (processed_next_state == processed_state).all():
            continue

        agent1.temporary_memory.push(processed_state, a1, processed_next_state, r1)
        agent2.temporary_memory.push(processed_state, a2, processed_next_state, r2)

        if agent1.temporary_memory.__len__() >= agent1.config.h:
            store_transition_in_buffer(agent1.temporary_memory, agent1.config.h, agent1.memory)
        if agent2.temporary_memory.__len__() >= agent2.config.h:
            store_transition_in_buffer(agent2.temporary_memory, agent2.config.h, agent2.memory)

        state = next_state
        processed_state = processed_next_state

    if agent1.policy.epsilon > agent1.config.min_epsilon:
        agent1.policy.epsilon -= (1 - agent1.config.min_epsilon) / total_episodes
    else:
        agent1.policy.epsilon = agent1.config.min_epsilon
    if agent2.policy.epsilon > agent2.config.min_epsilon:
        agent2.policy.epsilon -= (1 - agent2.config.min_epsilon) / total_episodes
    else:
        agent2.policy.epsilon = agent2.config.min_epsilon

def set_hyper_parameter(config: object):
    config.n_episodes = N_STEPS
    config.learning_rate = LEARNING_RATE
    config.min_epsilon = MIN_EPSILON
    config.epsilon_decay = EPSILON_DECAY
    return config

def initialize_agent(name: str, config: object, input_size: int = INPUT_IMAGE_DIM, hidden_size: int = HIDDEN_SIZE):
    agent = construct_agent(name, config)
    agent.policy_net = DuelDQN(input_size=input_size, out_size=OUT_SIZE, hidden_size=hidden_size,
                                batch_size=agent.config.batch_size, time_step=agent.config.h).to(device)
    agent.target_net = DuelDQN(input_size=input_size, out_size=OUT_SIZE, hidden_size=hidden_size,
                                batch_size=agent.config.batch_size, time_step=agent.config.h).to(device)
    print(f'{agent.name}: {agent.policy_net}')
    agent.target_net.load_state_dict(agent.policy_net.state_dict())
    agent.criterion = nn.SmoothL1Loss()
    agent.optimizer = torch.optim.Adam(agent.policy_net.parameters(), lr=agent.config.learning_rate)
    agent.memory = ReplayBuffer(MEMORY_SIZE)
    agent.temporary_memory = ReplayBuffer(agent.config.h)
    agent.updating_times = 0
    agent.loss = []
    return agent

def preprocess_image(image):
    return rgb2gray(image)

def synchronize_position(agent: object, env: object):
    for obj in env.objects:
        if obj.name == agent.name:
            agent.x = obj.x
            agent.y = obj.y

def fill_memory(agent1: object, agent2: object, env: object, config: object):
    # Fill Memory
    agent1.memory.clean()
    agent2.memory.clean()
    agent1.temporary_memory.clean()
    agent2.temporary_memory.clean()
    state = env.reset()
    processed_state = preprocess_image(state)
    synchronize_position(agent1, env)
    synchronize_position(agent2, env)

    for i in range(0, MEMORY_SIZE+config.h):

        a1 = np.random.randint(0, 4)
        a2 = np.random.randint(0, 4)
        r1, r2, next_state = env.step(agent1, agent2, a1, a2)
        processed_next_state = preprocess_image(next_state)
        agent1.temporary_memory.push(processed_state, a1, processed_next_state, r1)
        agent2.temporary_memory.push(processed_state, a2, processed_next_state, r2)

        if agent1.temporary_memory.__len__() >= agent1.config.h:
            store_transition_in_buffer(agent1.temporary_memory, agent1.config.h, agent1.memory)
        if agent2.temporary_memory.__len__() >= agent2.config.h:
            store_transition_in_buffer(agent2.temporary_memory, agent2.config.h, agent2.memory)

        state = next_state
        processed_state = processed_next_state

def store_transition_in_buffer(temporary_memory: object, time_step: int, new_memory: object):
    state, next_state = [], []
    for i in range(temporary_memory.__len__()-time_step, temporary_memory.__len__()):
        state.append(temporary_memory.memory[i].state)
        next_state.append(temporary_memory.memory[i].next_state)
    state, action, reward, next_state = np.array(state), temporary_memory.memory[-1].action, temporary_memory.memory[
        -1].reward, np.array(next_state)
    new_memory.push(state, action, next_state, reward)

def get_batch(agent: object):
    transitions = agent.memory.sample(agent.config.batch_size)
    state, action, next_state, reward = zip(*transitions)
    state = torch.from_numpy(np.array(state)).float().to(device)
    next_state = torch.from_numpy(np.array(next_state)).float().to(device)
    action = torch.from_numpy(np.array(action)).long().to(device)[:, None]
    reward = torch.from_numpy(np.array(reward)).float().to(device)[:, None]

    batch = Memory.Transition(state, action, next_state, reward)
    return batch

def optimize(agent: object):
    # update
    hidden_batch, cell_batch = agent.policy_net.init_hidden_states(agent.config.batch_size)

    batch = get_batch(agent)

    loss = agent.policy_net.train(agent, batch, hidden_batch, cell_batch)

    agent.updating_times += 1

def test_policy(agent1: object, agent2: object, env: object, seed: int = 110, steps: int = 1000):
    seed_everything(seed)
    frames = []
    agent1_reward = []
    agent2_reward = []
    obj_cnt = []

    state = env.reset()
    frames.append(state)
    obj_cnt.append(len(env.objects))

    synchronize_position(agent1, env)
    synchronize_position(agent2, env)
    agent1.reset()
    agent2.reset()
    agent1.policy.epsilon = 0
    agent2.policy.epsilon = 0

    processed_state = preprocess_image(state)
    step_count = 0
    agent1_hs, agent1_cs = agent1.policy_net.init_hidden_states(batch_size=1)
    agent2_hs, agent2_cs = agent2.policy_net.init_hidden_states(batch_size=1)

    while step_count < steps:
        step_count += 1
        with torch.no_grad():
            s = torch.from_numpy(processed_state).float().to(device)
            agent1_q_vals, (agent1_hs, agent1_cs) = agent1.policy_net(s, batch_size=1, time_step=1,
                                                                      hidden_state=agent1_hs, cell_state=agent1_cs)
            agent2_q_vals, (agent2_hs, agent2_cs) = agent2.policy_net(s, batch_size=1, time_step=1,
                                                                      hidden_state=agent2_hs, cell_state=agent2_cs)

        a1 = agent1.policy.sample_action(None, agent1_q_vals)
        a2 = agent2.policy.sample_action(None, agent2_q_vals)

        r1, r2, next_state = env.step(agent1, agent2, a1, a2)
        processed_next_state = preprocess_image(next_state)
        agent1.update(r1, a1, a2)
        agent2.update(r2, a2, a1)
        env.update(r1 + r2)
        frames.append(next_state)
        obj_cnt.append(len(env.objects))
        agent1_reward.append(r1)
        agent2_reward.append(r2)

        state = next_state
        processed_state = processed_next_state

    print(f'Total Reward: Agent 1: {sum(agent1_reward)}, Agent 2: {sum(agent2_reward)}')
    return frames, agent1_reward, agent2_reward, obj_cnt

def show_result(agent1_reward: list, agent2_reward: list):
    CC, CD, DC, DD, ND, DN, NN, NC, CN = 0, 0, 0, 0 ,0 ,0 ,0, 0, 0

    for i in range(len(agent1_reward)):

        if agent1_reward[i] in [0.8,1] and agent2_reward[i] in [0.8,1]:
            CC += 1
        elif agent1_reward[i] in [0.3,0.5] and agent2_reward[i] in [0.3,0.5]:
            CD += 1
        elif agent1_reward[i] == -1 and agent2_reward[i] == -1:
            DD += 1
        elif agent1_reward[i] in [-2.2,-2] and agent2_reward[i] in [0.8,1]:
            ND += 1
        elif agent1_reward[i] in [0.8,1] and agent2_reward[i] in [-2.2,-2]:
            DN += 1
        elif agent1_reward[i] in [0.8,1] and agent2_reward[i] in [-0.1,-0.2]:
            CN += 1
        elif agent1_reward[i] in [-0.1,-0.2] and agent2_reward[i] in [0.8,1]:
            NC += 1
        elif agent1_reward[i] in [-0.1,-0.2] and agent2_reward[i] in [-0.1,-0.2]:
            NN += 1
        else:
            print(agent1_reward[i], agent2_reward[i])
        DC = CD
    print(f'CC: {CC}, CD+DC: {CD}, DD: {DD}, ND: {ND}, DN: {DN}, NN: {NN}, NC: {NC}, CN: {CN}')
    print(f'agent1: total N: {ND+NN+NC}, total C: {CC+CD+CN}, total D: {DN+DD}' )
    print(f'agent2: total N: {DN+NN+CN}, total C: {CC+DC+NC}, total D: {ND+DD}' )
    return CC, CD, DC, DD, ND, DN, NN, NC, CN