import torch
import torch.nn as nn
import random
import numpy as np
from utils import *
from model import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MaxminDQN(DQN):

    @staticmethod
    def maxmin_q_vals(net_dict: dict[int, object], state: Type.TensorType):
        # maxmin q learning
        q_min = net_dict[list(net_dict.keys())[0]](state).clone()
        for m in net_dict:
            q = net_dict[m](state)
            q_min = torch.min(q_min, q)
        return q_min

    @staticmethod
    def compute_targets(target_net_dict: dict[int, object], rewards: Type.TensorType, next_states: Type.TensorType, discount: float):
        with torch.no_grad():
            q_min = MaxminDQN.maxmin_q_vals(target_net_dict, next_states)
            q_next = q_min.max(1)[0]
            q_target = rewards + discount * q_next[:, None]
        return q_target

    @staticmethod
    def optimize(policy_net: object, target_net_dict: dict[int, object], optimizer: object, batch: object, discount: float, criterion):
        state, action, reward, next_state = batch.state, batch.action, batch.reward, batch.next_state
        # compute the q value
        q_val = MaxminDQN.compute_q_vals(policy_net, state, action)
        with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
            target = MaxminDQN.compute_targets(target_net_dict, reward, next_state, discount)
        loss = criterion(q_val, target)
        # print(action[:5],reward[:5],q_val[:5],target[:5],loss)

        # backpropagation of loss to Neural Network (PyTorch magic)
        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        # for param in agent.policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)  # DQN gradient clipping: Clamps all elements in input into the range [ min, max ].
        optimizer.step()
        return loss

    @staticmethod
    def get_action(policy_net_dict: dict[int, object], state: Type.TensorType, n_actions: int, policy: object) -> int:
        if state is None:
            return random.randint(0, n_actions - 1)

        if not list(policy_net_dict.keys()):
            return random.randint(0, n_actions - 1)

        if not isinstance(state, tuple):
            state = state[None]
        else:
            state = (state[0][None], state[1][None])  # used by LSTMVariant network

        q_min = MaxminDQN.maxmin_q_vals(policy_net_dict, state).cpu().detach().numpy().flatten()
        action = policy.sample_action(state, q_min)
        return action