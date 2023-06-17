import torch
import numpy as np
import itertools
from PIL import Image
import matplotlib.pyplot as plt
import random


class GameObject(object):
    def __init__(self, coordinates, size, intensity, channel, name):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.intensity = intensity
        self.channel = channel
        self.name = name


class Coins(object):

    def __init__(self, size: int, config: object, spawn_prob: float = 0.005, image_dim: int = 84):
        self.size_x = size
        self.size_y = size
        self.n_actions = 4
        self.image_dim = image_dim
        self.spawn_prob = spawn_prob
        self.running_score = 0.0
        self.config = config
        a = self.reset()
        plt.imshow(np.clip(a,0,1), interpolation="nearest")
        # plt.imshow((a * 255), interpolation="nearest")
        plt.savefig('imshow.png')


    def reset(self):
        self.running_score = 0.0
        self.objects = []
        agent1 = GameObject(self.new_position(), 1, 1, 0, 'agent1')
        self.objects.append(agent1)
        agent2 = GameObject(self.new_position(), 1, 1, 1, 'agent2')
        self.objects.append(agent2)
        coin1 = GameObject(self.new_position(), 1, 1, 0, 'coin1')
        self.objects.append(coin1)
        coin2 = GameObject(self.new_position(), 1, 1, 1, 'coin2')
        self.objects.append(coin2)
        state = self.render_env()
        return state

    def new_position(self):
        iterables = [range(self.size_x), range(self.size_y)]
        points = []
        for t in itertools.product(*iterables):
            points.append(t)
        current_positions = []
        for obj in self.objects:
            if (obj.x, obj.y) not in current_positions:
                current_positions.append((obj.x, obj.y))
        for pos in current_positions:
            points.remove(pos)
        loc = np.random.choice(range(len(points)),replace=False)
        return points[loc]

    def move(self, agent: object, action: int):
        """
        action: north->0, south->1, west->2, east->3
        return next position
        """
        if action == 0:
            agent.y -= 1
        if action == 1:
            agent.y += 1
        if action == 2:
            agent.x -= 1
        if action == 3:
            agent.x += 1
        agent.x = np.clip(agent.x, 0, self.size_x - 1)
        agent.y = np.clip(agent.y, 0, self.size_y - 1)

    def step(self, agent1: object, agent2: object, a1: int, a2: int):
        r1, r2 = 0, 0
        # move agents
        self.move(agent1, a1)
        self.move(agent2, a2)
        for key, obj in enumerate(self.objects):
            if agent1.name == obj.name:
                if agent1.x == obj.x and agent1.y == obj.y:
                    r1 -= 0.2
                obj.x, obj.y = agent1.x, agent1.y
            if agent2.name == obj.name:
                if agent2.x == obj.x and agent2.y == obj.y:
                    r2 -= 0.2
                obj.x, obj.y = agent2.x, agent2.y
        # check goals
        if agent1.x == agent2.x and agent1.y == agent2.y:
            for key, obj in enumerate(self.objects):
                if 'coin' in obj.name and agent1.x == obj.x and agent1.y == obj.y:
                    r1, r2 = 0.5, 0.5
                    # print(f' 0  {obj.x}, {obj.y}, {agent1.x}, {agent1.y}, {agent2.x}, {agent2.y} ')
                    self.objects[key] = GameObject(self.new_position(), obj.size, obj.intensity, obj.channel, obj.name)
                    # self.objects.insert(key,
                    #                     GameObject(self.new_position(), obj.size, obj.intensity, obj.channel, obj.name))
                    # self.objects.remove(obj)

        for key, obj in enumerate(self.objects):
            if 'coin' in obj.name and agent1.x == obj.x and agent1.y == obj.y:
                r1 += 1
                if obj.name == 'coin2':
                    r2 += -2
                self.objects[key] = GameObject(self.new_position(), obj.size, obj.intensity, obj.channel, obj.name)

            if 'coin' in obj.name and agent2.x == obj.x and agent2.y == obj.y:
                r2 += 1
                if obj.name == 'coin1':
                    r1 += -2
                self.objects[key] = GameObject(self.new_position(), obj.size, obj.intensity, obj.channel, obj.name)

        if r1 == 0:
            r1 = -0.1
        if r2 == 0:
            r2 = -0.1
        # self.spawn_coin()
        state = self.render_env()
        return r1, r2, state

    def spawn_coin(self):
        iterables = [range(self.size_x), range(self.size_y)]
        points = []
        for t in itertools.product(*iterables):
            points.append(t)
        current_positions = []
        for obj in self.objects:
            if (obj.x, obj.y) not in current_positions:
                current_positions.append((obj.x, obj.y))
        for pos in current_positions:
            points.remove(pos)

        for point in points:
            if np.random.rand(1) < self.spawn_prob:
                idx = random.randint(1, 2)
                self.objects.append(GameObject(point, 1, 1, idx-1, f'coin{idx}'))

    def update(self, reward: int):
        self.running_score = reward + self.config.discount * self.running_score

    def render_env(self):
        a = np.ones([self.size_y+2, self.size_x+2, 3])
        a[1:-1, 1:-1, :] = 0
        # a = np.zeros([self.size_y, self.size_x, 3])
        # for obj in self.objects:
        #     a[obj.y : obj.y+obj.size, obj.x : obj.x+obj.size, obj.channel] = obj.intensity
        #     if 'coin' in obj.name:
        #         a[obj.y: obj.y + obj.size, obj.x: obj.x + obj.size, 0] = 1
        for obj in self.objects:
            a[obj.y+1 : obj.y+obj.size+1, obj.x+1 : obj.x+obj.size+1, obj.channel] = obj.intensity
            if 'coin' in obj.name:
                a[obj.y+1: obj.y+obj.size+1, obj.x+1: obj.x+obj.size+1, 2] = 1
        b = np.array(Image.fromarray(a[:, :, 0]).resize(size=(self.image_dim, self.image_dim), resample=Image.Resampling.NEAREST))
        c = np.array(Image.fromarray(a[:, :, 1]).resize(size=(self.image_dim, self.image_dim), resample=Image.Resampling.NEAREST))
        d = np.array(Image.fromarray(a[:, :, 2]).resize(size=(self.image_dim, self.image_dim), resample=Image.Resampling.NEAREST))

        a = np.stack([b, c, d], axis=2)
        return a

# env = Coins(size=5, config=None)










class GridWorld():
    """
    -------------
    |   |   |   |
    -------------
    | A |   | B |
    -------------
    |   |   |   |
    -------------

    """
    def __init__(self, config: object, board_rows: int = 3, board_cols: int = 3):
        self.config = config
        self.episode = 0
        self.running_score = 0.0
        self.board_rows = board_rows
        self.board_cols = board_cols
        self.board = torch.zeros([board_rows, board_cols])
        self.position_a = None
        self.position_b = None
        print('======= Initialize the environment: Grid World =======')
        self.initialize_position()

    def initialize_position(self, **kwargs):
        # intialize_position
        for key, arg in kwargs.items():
            if "a" in key or "A" in key:
                self.position_a = list(arg)
            if "b" in key or "B" in key:
                self.position_b = list(arg)
        if self.position_a is None:
            self.position_a = [int(self.board_rows/2), 0]
        if self.position_b is None:
            self.position_b = [int(self.board_rows/2),self.board_cols-1]
        self.finish_position_a = self.position_b
        self.finish_position_b = self.position_a

    def next_position(self, position, action):
        """
        action: north->0, south->1, west->2, east->3
        return next position
        """
        if action == 0:
            position[0] -= 1
        if action == 1:
            position[0] += 1
        if action == 2:
            position[1] -= 1
        if action == 3:
            position[1] += 1
        position[0] = np.clip(position[0], 0, self.board_rows-1)
        position[1] = np.clip(position[1], 0, self.board_cols-1)
        return position

    def step(self, a1, a2):
        done = False
        self.position_a = self.next_position(self.position_a, a1)
        self.position_b = self.next_position(self.position_b, a2)
        if self.position_a == self.finish_position_a and self.position_b == self.finish_position_b:
            done = True
            r1, r2 = 1, 1
        elif self.position_a == self.finish_position_a and self.position_b != self.finish_position_b:
            r1, r2 = 2, -1
        elif self.position_a != self.finish_position_a and self.position_b == self.finish_position_b:
            r1, r2 = -1, 2
        if self.position_a == self.position_b:
            done = True
            r1, r2 = 0, 0
        if not done:
            r1, r2 = -0.1, -0.1
        if done:
            self.initialize_position()
        return r1, r2

    def showBoard(self):
        for i in range(0, self.board_rows):
            print('-------------')
            out = '| '
            for j in range(0, self.board_rows):
                if self.position_a == [i, j]:
                    token = 'A'
                elif self.position_b == [i, j]:
                    token = 'B'
                else:
                    token = ' '
                out += token + ' | '
            print(out)
        print('-------------')


# grid = GridWorld(None)
# grid.showBoard()






