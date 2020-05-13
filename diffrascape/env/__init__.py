import gym
import numpy as np

from tensorforce.environments import Environment


class ShortWalk(gym.Env):
    """
    grid_world arrays have a redundant 3rd dimension with shape 1 which is needed for keras 2D convolutional layers
    """

    action_space = gym.spaces.Discrete(n=4)

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


def get_val(mean, var):
    return mean + var * 2.0 * (np.random.random() - 0.5)


class BadSeeds(Environment):
    def __init__(
        self,
        clist=[10.0, 10.0, 10.0],
        vlist=[3.0, 4.0, 5.0],
        max_turns=30,
        bad_list=[False, False, True],
    ):
        super().__init__()
        self.N = len(clist)
        self.clist = clist
        self.vlist = vlist
        self.max_turns = max_turns
        self.turn = 0
        self.picked_count = np.zeros(self.N)
        self.measured_list = []
        self.bad_list = bad_list

        self.reset()

    def current_mean(self):

        return [np.mean([el for el in sublist]) for sublist in self.measured_list]

        # return mean_list

    def current_stddev(self):
        return [np.var([el for el in sublist]) ** 0.5 for sublist in self.measured_list]

    def current_norm_stddev(self):
        stddev_list = self.current_stddev()
        return stddev_list / max(stddev_list)

    def states(self):
        return dict(type="float", shape=(self.N * 2,))

    def actions(self):
        return dict(type="int", num_values=self.N)

    def reset(self):
        # rshuffle the clist/vlist order, but don't change them.
        z = list(zip(self.clist, self.vlist, self.bad_list))
        np.random.shuffle(z)
        self.clist, self.vlist, self.bad_list = zip(*z)
        self.measured_list = []
        self.turn = 0
        self.picked_count = np.zeros(self.N)
        
        # take first measurments
        for i in range(self.N):
            this_list = []
            for j in range(2):
                this_list.append(get_val(mean=self.clist[i], var=self.vlist[i]))
            self.measured_list.append(this_list)

        # state is the current variance of each point
        self.state = np.zeros(2 * self.N)
        for i in range(self.N):
            self.state[i] = np.var(self.measured_list[i])  # **.5
            self.state[int(self.N) + i] = self.picked_count[i] / float(self.max_turns)

        return self.state

    def execute(self, actions):
        # assert 0 <= actions.item() <= 3

        # take another measurement of value 'action'
        this_val = get_val(mean=self.clist[actions], var=self.vlist[actions])
        self.picked_count[int(actions)] += 1.0
        self.measured_list[actions].append(this_val)

        next_state = self.state
        next_state[actions] = np.var(self.measured_list[actions])  # **.5
        next_state[self.N + actions] = (
            self.picked_count[(int(actions))] / self.max_turns
        )

        terminal = False

        reward = 0
        self.turn += 1
        if self.bad_list[actions]:  # if this is a bad sample
            reward += 1  # give 1 point

        if self.turn >= self.max_turns:
            terminal = True
            reward = 0.0
            # check if we've gotten min score on bad points
            for i in range(self.N):
                if self.bad_list[i] and self.picked_count[i] >= 20:
                    reward += 100
                    # print ('woohoo '+str(i))

        return next_state, terminal, reward


class BadSeedsTheSequel(Environment):
    def __init__(
            self,
            centers=[10.0, 10.0, 10.0],
            variances=[3.0, 4.0, 5.0],
            max_turns=30,
            bad_seeds=[False, False, True],
            bad_seed_min_picks=5,
    ):
        super().__init__()
        self.N = len(centers)
        self.centers = centers
        self.variances = variances
        self.max_turns = max_turns
        self.turn = 0
        self.picked_count = np.zeros(self.N)
        self.measured_list = []
        self.bad_seeds = bad_seeds
        self.bad_seed_min_picks = bad_seed_min_picks

        self.reset()

    def current_mean(self):

        return [np.mean([el for el in sublist]) for sublist in self.measured_list]

        # return mean_list

    def current_stddev(self):
        return [np.var([el for el in sublist]) ** 0.5 for sublist in self.measured_list]

    def current_norm_stddev(self):
        stddev_list = self.current_stddev()
        return stddev_list / max(stddev_list)

    def states(self):
        #return dict(type="float", shape=(self.N * 2,))
        return dict(type="float", shape=(self.max_turns, self.N))

    def actions(self):
        return dict(type="int", num_values=self.N)

    def reset(self):
        # rshuffle the clist/vlist order, but don't change them.
        z = list(zip(self.centers, self.variances, self.bad_seeds))
        np.random.shuffle(z)
        self.centers, self.variances, self.bad_seeds = zip(*z)
        self.measured_list = []
        self.turn = 0
        self.picked_count = np.zeros(self.N)

        # max_turns x N
        self.state = np.zeros((self.max_turns, self.N))
        return self.state

    def execute(self, actions):
        # assert 0 <= actions.item() <= 3

        # take another measurement of value 'action'
        this_val = get_val(mean=self.centers[actions], var=self.variances[actions])
        self.picked_count[int(actions)] += 1.0
        #self.measured_list[actions].append(this_val)
        self.state[self.turn, int(actions)] = this_val

        next_state = self.state
        # next_state[actions] = np.var(self.measured_list[actions])  # **.5
        # next_state[self.N + actions] = (
        #         self.picked_count[(int(actions))] / self.max_turns
        # )

        terminal = False

        reward = 0
        self.turn += 1
        # tried rewarding every choice
        # tried rewarding only at the end - learning stopped and I don't know why
        # trying rewarding after 5 picks - 101230
        # try rewarding after 10 picks
        if self.bad_seeds[actions] and self.picked_count[actions] >= self.bad_seed_min_picks:  # if this is a bad seed
            reward += 1  # give 1 point

        if self.turn >= self.max_turns:
            terminal = True
            reward = 0.0
            # check if we've gotten min score on bad points
            for i in range(self.N):
                if self.bad_seeds[i] and self.picked_count[i] >= 20:
                    reward += 100
                    # print ('woohoo '+str(i))

        return next_state, terminal, reward
