import gym
import numpy as np

from diffrascape.rl.deep_q_network import ExperienceHistory


def test_experience_history():
    np.random.seed(0)

    # six states for 5 actions
    all_states = np.random.rand(6, 2)
    all_actions = np.random.randint(low=0, high=3, size=5)
    all_rewards = np.array([0.0, 1.0, 0.0, 0.5, 1.0])
    all_done = np.array([False, False, False, True, False])

    h = ExperienceHistory(state_shape=(all_states.shape[1], ), history_size=5, batch_size=3)

    for i in range(5):
        state0 = all_states[i]
        action = all_actions[i]
        reward = all_rewards[i]
        done = all_done[i]
        state1 = all_states[i+1]
        h.append(state0=state0, action=action, reward=reward, done=done, state1=state1)

        print(f"state0: {state0}")
        print(f"action: {action}")
        print(f"reward: {reward}")
        print(f"done  : {done}")
        print(f"state1: {state1}")
        print()

    assert len(h) == 5

    state0_batch, action_batch, reward_batch, done_batch, state1_batch = h.sample_batch()

    assert state0_batch.shape == (3, 2)
    assert state0_batch.dtype == np.float64
    assert action_batch.shape == (3, )
    assert action_batch.dtype == np.int
    assert reward_batch.shape == (3, )
    assert reward_batch.dtype == np.float64
    assert done_batch.shape == (3, )
    assert done_batch.dtype == np.bool
    assert state1_batch.shape == (3, 2)
    assert state1_batch.dtype == np.float64

    print(state0_batch)
    print(action_batch)
    print(reward_batch)
    print(done_batch)
    print(state1_batch)

