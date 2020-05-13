import numpy as np

import gym
import gym.spaces as spaces


def get_val(mean, stddev):
    return np.random.normal(loc=mean, scale=stddev)


class Variance1(gym.Env):
    """ my new game """

    def __init__(self, N=10, vmax=10, stddev_max=10, stddev_min=1, max_turns=100):

        self.N = N

        self.vmax = vmax

        self.stddev_max = stddev_max

        self.stddev_min = stddev_min

        self.max_turns = max_turns

        self.turn = 0

        self.mean_list = np.random.random(self.N) * self.vmax

        self.stddev_list = np.random.random(self.N) * (self.stddev_max - self.stddev_min) + self.stddev_min

        # self.stddev_list = np.linspace(0.1,1,10)

        self.measured_list = []

        # take first measurments

        for i in range(self.N):

            this_list = []

            for j in range(2):
                this_list.append(get_val(mean=self.mean_list[i], stddev=self.stddev_list[i]))

            self.measured_list.append(this_list)

        # at this point, each dataset has 2 initial measurments

        self.action_space = spaces.Discrete(N)  # assume we can select actions 0 to N

        # setup observation_space

        high = np.array((self.N) * [np.finfo(np.float32).max])

        low = np.array((self.N) * [0.0])

        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        ##############

        #self.seed()

        self.reset()

        # state is the current variance of each point

        # self.state = np.zeros(self.N)

        # for i in range(self.N):

        #    self.state[i] = np.var(self.measured_list[i])**.5

    # def seed(self, seed=None):
    #
    #     self.np_random, seed = seeding.np_random(seed)
    #
    #     return [seed]

    def step(self, action):

        assert self.action_space.contains(action)

        # take another measurement of value 'action'

        this_val = get_val(mean=self.mean_list[action], stddev=self.stddev_list[action])

        self.measured_list[action].append(this_val)

        self.state[action] = np.var(self.measured_list[action]) ** .5

        done = False

        reward = 0

        self.turn += 1

        if self.turn >= self.max_turns:

            done = True

            reward = 10.0

            for i in range(self.N):
                this_mean = np.mean(self.measured_list[i])

                this_true = self.mean_list[i]

                reward -= abs(this_mean - this_true) ** 2

        return self.state, reward, done, {}

    def reset(self):
        self.turn = 0

        self.mean_list = np.random.random(self.N) * self.vmax

        self.stddev_list = np.random.random(self.N) * (self.stddev_max - self.stddev_min) + self.stddev_min

        self.measured_list = []

        # take first measurments

        for i in range(self.N):

            this_list = []

            for j in range(2):
                this_list.append(get_val(mean=self.mean_list[i], stddev=self.stddev_list[i]))

            self.measured_list.append(this_list)

        # state is the current variance of each point

        self.state = np.zeros(self.N)

        for i in range(self.N):
            self.state[i] = np.var(self.measured_list[i]) ** .5

        return self.state