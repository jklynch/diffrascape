import numpy as np

from diffrascape.env import ShortWalk


def test_short_walk():
    short_walk_env = ShortWalk(width=3, height=4)
    short_walk_env.reset()
    assert (
        short_walk_env.get_current_grid_value(
            w=short_walk_env.current_point[0], h=short_walk_env.current_point[1]
        )
        == -10.0
    )

    first_point = short_walk_env.current_point

    state, reward, done, _ = short_walk_env.step(action=0)
    assert np.any(first_point != short_walk_env.current_point)
    assert np.all(
        short_walk_env.current_point == (first_point + short_walk_env.action_table[0])
    )
