import matplotlib.pyplot as plt
import numpy as np

from diffrascape.env import ShortWalk


def plot_grid_world(grid_world):
    fig = plt.figure()
    plt.imshow(grid_world)
    plt.colorbar()


np.random.seed(0)

short_walk_env = ShortWalk(width=5, height=6)
short_walk_env.reset()
plot_grid_world(short_walk_env.grid_world)

state1, reward1, done1, _ = short_walk_env.step(action=0)
plot_grid_world(state1)
state2, reward2, done2, _ = short_walk_env.step(action=0)
plot_grid_world(state2)
plt.show()
