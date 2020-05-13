import gym
import numpy as np


class KArm(gym.Env):
    """
    """
    k = 10
    action_space = gym.spaces.Discrete(n=k)
    observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(k, 1)), dtype=np.fload

    def __init__(self, width, height):
        self.observation_space = gym.spaces.Box(
            low=-10.0, high=5.0, shape=(width, height, 1), dtype=np.float
        )

        self.width = width
        self.height = height

        self.grid_world = None
        self.start_point = None
        self.current_point = None

        self.action_table = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])

    def reset(self):
        self.grid_world = np.random.choice(
            a=[-5.0, 0.0, 5.0], size=(self.width, self.height, 1),
        )
        self.start_point = (self.width // 2, self.height // 2)
        self.grid_world[self.start_point[0], self.start_point[1], 0] = -10.0
        self.current_point = self.start_point
        return self.grid_world

    def step(self, action):
        """
        Parameters
        ----------
        action: int
            0 - current_point + (-1,  0)
            1 - current_point + ( 1,  0)
            2 - current_point + ( 0, -1)
            3 - current_point + ( 0,  1)
        """
        assert 0 <= action <= 3

        self.current_point += self.action_table[action]
        if (
            0 < self.current_point[0] < self.width
            and 0 < self.current_point[1] < self.height
        ):
            reward = (
                -1.0 + self.grid_world[self.current_point[0], self.current_point[1], 0]
            )
            done = False
            self.grid_world[self.current_point[0], self.current_point[1], 0] = -10.0
        else:
            reward = 1.0
            done = True

        return self.grid_world, reward, done, {}

    def get_current_grid_value(self, w, h):
        return self.grid_world[w, h]
